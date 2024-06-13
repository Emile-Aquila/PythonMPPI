import numpy as np
from models.robot_model import RobotState
from models.veltypes import VOmega
from models.dynamics_model import ParallelTwoWheelVehicleModel


class MPPIPlanner:
    def __init__(self, model: ParallelTwoWheelVehicleModel, horizon: int, num_samples: int, lambda_: float,
                 sigma_v: float, sigma_omega):
        self.model: ParallelTwoWheelVehicleModel = model
        self.horizon: int = horizon
        self.num_samples: int = num_samples
        self.lambda_: float = lambda_
        self.rng: np.random.Generator = np.random.default_rng()
        self.sigma_v: float = sigma_v
        self.sigma_omega: float = sigma_omega

        self.act_prev: VOmega = VOmega(0.0, 0.0)
        self._act_spec_size: int = self.act_prev.size
        self.input_traj_prev: np.ndarray = np.zeros((self.horizon, self._act_spec_size))
        self.sampled_trajs: np.ndarray = np.zeros((self.num_samples, self.horizon, 5))
        self.goal: np.ndarray = np.array([0.0, 0.0, 0.0])

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    @staticmethod
    def terminal_cost(state: np.ndarray, goal: np.ndarray) -> float:
        diff = state[:3] - goal
        diff[2] /= 2.0
        return np.linalg.norm(diff)

    @staticmethod
    def stage_cost(state: np.ndarray, goal: np.ndarray) -> float:
        diff = state[:3] - goal
        diff[2] /= 2.0
        return np.linalg.norm(diff)

    def _rollout(self, first_state: np.ndarray, base_acts: np.ndarray) -> np.ndarray:
        trajectory: np.ndarray = first_state.reshape(1, -1)
        inputs: np.ndarray = self.rng.multivariate_normal(
            mean=np.zeros(self._act_spec_size), cov=np.diag([self.sigma_v, self.sigma_omega]), size=self.horizon
        ) + base_acts

        for raw_input in inputs:
            input_pre = trajectory[-1][-self._act_spec_size:]
            tmp_input = self.model.constraints.clip_act_numpy(input_pre, raw_input)
            new_traj = self.model.kinematics_numpy(trajectory[-1], tmp_input, self.model.dt).reshape(1, -1)
            trajectory = np.append(trajectory, new_traj, axis=0)

        return trajectory[1:]

    def trajectory_costs(self, trajectories: np.ndarray) -> np.ndarray:
        costs = np.array(
            [[self.stage_cost(obs, self.goal) for obs in traj[0:-1]] + [self.terminal_cost(traj[-1], self.goal)] for traj in trajectories])
        sum_costs = np.sum(costs, axis=1)
        return sum_costs

    def policy(self, state: RobotState[VOmega]) -> VOmega:
        state_np = state.to_numpy()

        # sample trajectories
        trajs = np.array([self._rollout(state_np, self.input_traj_prev) for _ in range(self.num_samples)])
        input_trajs: np.ndarray = trajs[:, :, -self._act_spec_size:]
        self.sampled_trajs = trajs

        # calculate costs
        sum_costs = self.trajectory_costs(trajs)

        # importance sampling
        input_term = np.sum(
            np.sum(np.array([1 / self.sigma_v, 1 / self.sigma_omega]) * input_trajs * self.input_traj_prev, axis=2),
            axis=1)
        sum_costs = -self.lambda_ * sum_costs - input_term

        weights = np.exp(sum_costs - np.max(sum_costs))
        weights /= np.sum(weights)
        for i, weight in enumerate(weights):
            input_trajs[i] = weight * input_trajs[i]

        # calculate new input
        self.input_traj_prev = np.sum(input_trajs, axis=0)
        self.act_prev = self.model.constraints.clip_act(
            self.act_prev,
            VOmega(self.input_traj_prev[0, 0], self.input_traj_prev[0, 1])
        )
        return self.act_prev
