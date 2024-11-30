from typing import Tuple, Callable
import jax.numpy as jnp
import jax
from models.robot_model import RobotState
from models.veltypes import VOmega
from models.dynamics_model import ParallelTwoWheelVehicleModel
from functools import partial
from flax import struct


@struct.dataclass
class SamplingParams:
    lambda_: float
    sigma_v: float
    sigma_omega: float
    key: jax.random.PRNGKey
    act_spec_size: int = struct.field(pytree_node=False)
    num_samples: int = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)


@struct.dataclass
class SamplingData:
    sampled_state_traj: jnp.ndarray  # (num_samples, horizon, 5)
    input_traj: jnp.ndarray  # (horizon, 2)
    input: jnp.ndarray  # (2,)
    params: SamplingParams



class MPPIPlanner:
    def __init__(self, model: ParallelTwoWheelVehicleModel, stage_cost: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 terminal_cost: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> None:
        self.model: ParallelTwoWheelVehicleModel = model
        self.stage_cost: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = stage_cost
        self.terminal_cost: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = terminal_cost

        MPPIPlanner.stage_cost = staticmethod(stage_cost)
        MPPIPlanner.terminal_cost = staticmethod(terminal_cost)

        jax.config.update("jax_debug_nans", False)
        jax.config.update("jax_debug_infs", False)

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
    def _rollout(self, sub_key: jax.random.PRNGKey, first_state: jnp.ndarray, base_acts: jnp.ndarray,
                 mean: jnp.ndarray, cov: jnp.ndarray, params: SamplingParams) -> jnp.ndarray:
        def scan_fn(val, x):
            input_pre = val[-params.act_spec_size:]
            tmp_input = self.model.constraints.clip_act_jax(input_pre, x)
            new_traj = self.model.kinematic_jax(val, tmp_input, self.model.dt).reshape(1, -1)
            return new_traj[0], new_traj[0]

        inputs: jnp.ndarray = jax.random.multivariate_normal(sub_key, mean=mean, cov=cov, shape=(params.horizon,)) + base_acts
        _, trajectory = jax.lax.scan(scan_fn, first_state, inputs)
        return trajectory

    @partial(jax.jit, static_argnums=(0, -2))
    def _rollout_n(self, sub_key: jax.random.PRNGKey, first_state: jnp.ndarray, base_acts: jnp.ndarray, n: int, params: SamplingParams) -> jnp.ndarray:
        mean = jnp.zeros(base_acts.shape[-1])
        cov = jnp.diag(jnp.array([params.sigma_v, params.sigma_omega]))
        ans = jax.vmap(self._rollout, in_axes=(0, None, None, None, None, None))(jax.random.split(sub_key, n), first_state, base_acts, mean, cov, params)
        return ans

    @partial(jax.jit, static_argnums=(0,))
    def calc_trajectory_costs(self, trajectories: jnp.ndarray, goal: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cost_sums, costs = jax.vmap(lambda x: self.trajectory_cost(x, goal), in_axes=0)(trajectories)  # (num_samples,), (num_samples, horizon)
        return cost_sums, costs

    @partial(jax.jit, static_argnums=(0,))
    def policy_jax(self, state: jnp.ndarray, goal: jnp.ndarray, sampling_data: SamplingData) -> SamplingData:
        params = sampling_data.params

        # sample trajectories
        key, sub_key = jax.random.split(params.key)
        trajs = self._rollout_n(sub_key, state, sampling_data.input_traj, params.num_samples, params)
        input_trajs: jnp.ndarray = trajs[:, :, -params.act_spec_size:]

        # calculate costs
        cost_sums, costs  = self.calc_trajectory_costs(trajs, goal)

        # importance sampling
        input_term = jnp.sum(jnp.sum(jnp.array([1 / params.sigma_v, 1 / params.sigma_omega]) * input_trajs * sampling_data.input_traj, axis=2), axis=1)
        cost_sums = -params.lambda_ * cost_sums - input_term

        cost_max = jnp.max(cost_sums)
        weights = jax.vmap(lambda x: jnp.exp(x - cost_max))(cost_sums)
        weights /= jnp.sum(weights)
        input_trajs = jax.vmap(lambda x, y: x * y, in_axes=(0, 0))(input_trajs, weights)

        # calculate new input
        input_traj = jnp.sum(input_trajs, axis=0)
        input_ans = self.model.constraints.clip_act_jax(sampling_data.input, input_traj[0])

        # Sampling Data Update
        new_data = SamplingData(
            sampled_state_traj=trajs,
            input_traj=input_traj,
            input=input_ans,
            params=SamplingParams(
                lambda_=params.lambda_,
                sigma_v=params.sigma_v,
                sigma_omega=params.sigma_omega,
                key=key,
                act_spec_size=params.act_spec_size,
                num_samples=params.num_samples,
                horizon=params.horizon
            )
        )
        return new_data

    def policy(self, state: jnp.ndarray, goal: jnp.ndarray, sampling_data: SamplingData) -> Tuple[VOmega, SamplingData]:
        new_data = self.policy_jax(state, goal, sampling_data)
        return VOmega(new_data.input[0], new_data.input[1]), new_data
