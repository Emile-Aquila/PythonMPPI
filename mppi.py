import jax.numpy as jnp
import jax
from models.robot_model import RobotState
from models.veltypes import VOmega
from models.dynamics_model import ParallelTwoWheelVehicleModel
# from numba import jit
import ray


class MPPIPlanner:
    def __init__(self, model: ParallelTwoWheelVehicleModel, horizon: int, num_samples: int, lambda_: float,
                 sigma_v: float, sigma_omega: float, n_cpu: int):
        self.model: ParallelTwoWheelVehicleModel = model
        self.horizon: int = horizon
        self.num_samples: int = num_samples
        self.lambda_: float = lambda_
        self.key: jax.random.PRNGKey = jax.random.PRNGKey(seed=0)
        self.sigma_v: float = sigma_v
        self.sigma_omega: float = sigma_omega

        self.act_prev: VOmega = VOmega(0.0, 0.0)
        self._act_spec_size: int = self.act_prev.size
        self.input_traj_prev: jnp.ndarray = jnp.zeros((self.horizon, self._act_spec_size))
        self.sampled_trajs: jnp.ndarray = jnp.zeros((self.num_samples, self.horizon, 5))
        self.goal: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])

        self._mean = jnp.zeros(self._act_spec_size)
        self._cov = jnp.diag(jnp.array([self.sigma_v, self.sigma_omega]))

        self._n_cpu: int = n_cpu
        ray.init(num_cpus=n_cpu)
        jax.config.update('jax_platform_name', 'cpu')

    def set_goal(self, goal: jnp.ndarray):
        self.goal = goal

    @staticmethod
    @jax.jit
    def terminal_cost(state: jnp.ndarray, goal: jnp.ndarray):
        diff = state[:3] - goal
        diff = diff.at[2].set(diff[2] / 2.0)
        return jnp.sqrt(jnp.power(diff, 2).sum())

    @staticmethod
    # @jax.jit
    def stage_cost(state: jnp.ndarray, goal: jnp.ndarray):
        diff = state[:3] - goal
        diff = diff.at[2].set(diff[2] / 2.0)
        return jnp.sqrt(jnp.power(diff, 2).sum())

    @staticmethod
    @jax.jit
    def stage_costs(trajectories: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        diff = trajectories[:, :, :3] - goal
        # diff[:, :, 2] /= 2.0
        diff = diff.at[:, :, 2].set(diff[:, :, 2] / 2.0)
        return jnp.sqrt(jnp.power(diff, 2).sum(axis=-1)).sum(axis=-1)

    def _rollout(self, sub_key: jax.random.PRNGKey, first_state: jnp.ndarray, base_acts: jnp.ndarray) -> jnp.ndarray:
        trajectory: jnp.ndarray = first_state.reshape(1, -1)
        # inputs: jnp.ndarray = sub_key.multivariate_normal(mean=self._mean, cov=self._cov, size=self.horizon) + base_acts
        inputs: jnp.ndarray = jax.random.multivariate_normal(sub_key, mean=self._mean, cov=self._cov,
                                                             shape=(self.horizon,)) + base_acts

        for raw_input in inputs:
            input_pre = trajectory[-1][-self._act_spec_size:]
            tmp_input = self.model.constraints.clip_act_jax(input_pre, raw_input)
            new_traj = self.model.kinematic_jax(trajectory[-1], tmp_input, self.model.dt).reshape(1, -1)
            trajectory = jnp.append(trajectory, new_traj, axis=0)

        return trajectory[1:]

    @ray.remote
    def rollout_n(self, sub_key: jax.random.PRNGKey, first_state: jnp.ndarray, base_acts: jnp.ndarray,
                  n: int) -> jnp.ndarray:
        return jnp.array([self._rollout(sub_key, first_state, base_acts) for _ in range(n)])

    def trajectory_costs(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        costs = self.stage_costs(trajectories[:, 0:-1, :], self.goal)
        costs += jnp.array([self.terminal_cost(traj[-1], self.goal) for traj in trajectories])
        return costs

    def policy(self, state: RobotState[VOmega]) -> VOmega:
        state_np = jnp.array(state.to_numpy())

        # sample trajectories
        trajs = jnp.array(
            ray.get([self.rollout_n.remote(self, rng, state_np, self.input_traj_prev, self.num_samples // self._n_cpu)
                     for rng in jax.random.split(self.key, self._n_cpu)])
        )
        trajs = jnp.concatenate(trajs, axis=0)
        # trajs = jnp.array([self._rollout(self.key, state_np, self.input_traj_prev) for _ in range(self.num_samples)])
        input_trajs: jnp.ndarray = trajs[:, :, -self._act_spec_size:]
        self.sampled_trajs = trajs

        # calculate costs
        sum_costs = self.trajectory_costs(trajs)

        # importance sampling
        input_term = jnp.sum(
            jnp.sum(jnp.array([1 / self.sigma_v, 1 / self.sigma_omega]) * input_trajs * self.input_traj_prev, axis=2),
            axis=1)
        sum_costs = -self.lambda_ * sum_costs - input_term

        weights = jnp.exp(sum_costs - jnp.max(sum_costs))
        weights /= jnp.sum(weights)
        for i, weight in enumerate(weights):
            input_trajs = input_trajs.at[i].set(weight * input_trajs[i])

        # calculate new input
        self.input_traj_prev = jnp.sum(input_trajs, axis=0)
        self.act_prev = self.model.constraints.clip_act(
            self.act_prev,
            VOmega(self.input_traj_prev[0, 0], self.input_traj_prev[0, 1])
        )
        return self.act_prev
