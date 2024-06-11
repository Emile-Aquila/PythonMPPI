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
        self.input_traj_prev: np.ndarray = np.zeros((self.horizon, 2))
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
        vs: np.ndarray = self.rng.normal(loc=0.0, scale=self.sigma_v, size=self.horizon) + base_acts[:, 0]
        omegas: np.ndarray = self.rng.normal(loc=0.0, scale=self.sigma_omega, size=self.horizon) + base_acts[:, 1]

        for v, omega in zip(vs, omegas):
            v_omega_pre = trajectory[-1][3:]
            v_omega = self.model.constraints.clip_act_numpy(v_omega_pre, np.array([v, omega]))
            new_traj = self.model.kinematics_numpy(trajectory[-1], v_omega, self.model.dt).reshape(1, -1)
            trajectory = np.append(trajectory, new_traj, axis=0)

        return trajectory[1:]

    def policy(self, state: RobotState[VOmega]) -> VOmega:
        state_np = state.to_numpy()
        trajs = np.array([self._rollout(state_np, self.input_traj_prev) for _ in range(self.num_samples)])
        input_trajs: np.ndarray = trajs[:, :, 3:]
        self.sampled_trajs = trajs

        costs = np.array(
            [[self.stage_cost(obs, self.goal) for obs in traj[0:-1]] + [self.terminal_cost(traj[-1], self.goal)] for traj in trajs])
        sum_costs = np.sum(costs, axis=1)

        input_term = np.sum(
            np.sum(np.array([1 / self.sigma_v, 1 / self.sigma_omega]) * input_trajs * self.input_traj_prev, axis=2),
            axis=1)
        sum_costs = -self.lambda_ * sum_costs - input_term

        weights = np.exp(sum_costs - np.max(sum_costs))
        weights /= np.sum(weights)
        for i, weight in enumerate(weights):
            input_trajs[i] = weight * input_trajs[i]

        self.input_traj_prev = np.sum(input_trajs, axis=0)
        self.act_prev = self.model.constraints.clip_act(self.act_prev,
                                                        VOmega(self.input_traj_prev[0, 0], self.input_traj_prev[0, 1]))
        return self.act_prev
