from typing import Tuple
import jax.numpy as jnp
import jax
from models.robot_model import RobotState
from models.veltypes import VOmega
from models.dynamics_model import ParallelTwoWheelVehicleModel
from functools import partial


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

        self.act_prev: jnp.ndarray = jnp.array(VOmega(0.0, 0.0).to_numpy())
        self._act_spec_size: int = self.act_prev.size
        self.input_traj_prev: jnp.ndarray = jnp.zeros((self.horizon, self._act_spec_size))
        self.sampled_trajs: jnp.ndarray = jnp.zeros((self.num_samples, self.horizon, 5))
        self.goal: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])

        self._mean = jnp.zeros(self._act_spec_size)
        self._cov = jnp.diag(jnp.array([self.sigma_v, self.sigma_omega]))

        self._n_cpu: int = n_cpu
        jax.config.update("jax_debug_nans", False)
        jax.config.update("jax_debug_infs", False)

    def set_goal(self, goal: jnp.ndarray):
        self.goal = goal

    @staticmethod
    @jax.jit
    def terminal_cost(state: jnp.ndarray, goal: jnp.ndarray):
        diff = state[:3] - goal
        diff = diff.at[2].set(diff[2] / 2.0)
        return jnp.sqrt(jnp.power(diff, 2).sum())

    @staticmethod
    @jax.jit
    def stage_cost(state: jnp.ndarray, goal: jnp.ndarray):
        diff = state[:3] - goal
        diff = diff.at[2].set(diff[2] / 2.0)
        return jnp.sqrt(jnp.power(diff, 2).sum())

    @staticmethod
    @jax.jit
    def trajectory_cost(trajectory: jnp.ndarray, goal: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cost_sum, costs = jax.lax.scan(
            lambda val, input_x: (val + MPPIPlanner.stage_cost(input_x, goal), MPPIPlanner.stage_cost(input_x, goal)),
            0.0, trajectory[0:-1, :]
        )
        terminal_cost = MPPIPlanner.terminal_cost(trajectory[-1], goal)
        costs = jnp.concatenate([costs, jnp.array([terminal_cost])])
        return cost_sum + terminal_cost, costs

    @partial(jax.jit, static_argnums=(0,))
    def _rollout(self, sub_key: jax.random.PRNGKey, first_state: jnp.ndarray, base_acts: jnp.ndarray) -> jnp.ndarray:
        inputs: jnp.ndarray = jax.random.multivariate_normal(sub_key, mean=self._mean, cov=self._cov, shape=(self.horizon,)) + base_acts

        def scan_fn(val, x):
            input_pre = val[-self._act_spec_size:]
            tmp_input = self.model.constraints.clip_act_jax(input_pre, x)
            new_traj = self.model.kinematic_jax(val, tmp_input, self.model.dt).reshape(1, -1)
            return new_traj[0], new_traj[0]

        _, trajectory = jax.lax.scan(scan_fn, first_state, inputs)
        return trajectory

    @partial(jax.jit, static_argnums=(0, -1))
    def rollout_n(self, sub_key: jax.random.PRNGKey, first_state: jnp.ndarray, base_acts: jnp.ndarray, n: int) -> jnp.ndarray:
        ans = jax.vmap(self._rollout, in_axes=(0, None, None))(jax.random.split(sub_key, n), first_state, base_acts)
        return ans

    @partial(jax.jit, static_argnums=(0,))
    def calc_trajectory_costs(self, trajectories: jnp.ndarray, goal: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cost_sums, costs = jax.vmap(lambda x: self.trajectory_cost(x, goal), in_axes=0)(trajectories)  # (num_samples,), (num_samples, horizon)
        return cost_sums, costs

    @partial(jax.jit, static_argnums=(0,))
    def policy_jax(self, state: jnp.ndarray, goal: jnp.ndarray, act_prev: jnp.ndarray, input_traj_prev: jnp.ndarray, key: jax.random.PRNGKey) \
            -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        # sample trajectories
        key, sub_key = jax.random.split(key)
        trajs = self.rollout_n(sub_key, state, input_traj_prev, self.num_samples)
        input_trajs: jnp.ndarray = trajs[:, :, -self._act_spec_size:]

        # calculate costs
        cost_sums, costs  = self.calc_trajectory_costs(trajs, goal)

        # importance sampling
        input_term = jnp.sum(
            jnp.sum(jnp.array([1 / self.sigma_v, 1 / self.sigma_omega]) * input_trajs * input_traj_prev, axis=2), axis=1)
        cost_sums = -self.lambda_ * cost_sums - input_term

        cost_max = jnp.max(cost_sums)
        weights = jax.vmap(lambda x: jnp.exp(x - cost_max))(cost_sums)
        weights /= jnp.sum(weights)
        input_trajs = jax.vmap(lambda x, y: x * y, in_axes=(0, 0))(input_trajs, weights)

        # calculate new input
        input_trajs = jnp.sum(input_trajs, axis=0)
        act = self.model.constraints.clip_act_jax(act_prev, input_trajs[0])
        return act, input_trajs, trajs, key

    def policy(self, state: RobotState[VOmega]) -> VOmega:
        state_jax = jnp.array(state.to_numpy())
        self.act_prev, self.input_traj_prev, self.sampled_trajs, self.key = self.policy_jax(
            state_jax, self.goal, self.act_prev, self.input_traj_prev, self.key
        )
        return VOmega(self.act_prev[0], self.act_prev[1])
