import jax
import jax.numpy as jnp
import numpy as np
from models.robot_model import RobotState
from mppi import MPPIPlanner, SamplingData, SamplingParams
from objects.field import Point2D, Field, GenTestField, Circle
from models.veltypes import VOmega, VOmegaConstraints
from models.dynamics_model import ParallelTwoWheelVehicleModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import functools
import time


@jax.jit
def terminal_cost(state: jnp.ndarray, goal: jnp.ndarray):
    diff = state[:3] - goal
    diff = diff.at[2].set(diff[2] / 2.0)
    return jnp.sqrt(jnp.power(diff, 2).sum())


@jax.jit
def stage_cost(state: jnp.ndarray, goal: jnp.ndarray):
    diff = state[:3] - goal
    diff = diff.at[2].set(diff[2] / 2.0)
    return jnp.sqrt(jnp.power(diff, 2).sum())


def main():
    field: Field = GenTestField(0)
    dt = 0.1

    horizon = 40
    num_samples = 500
    act_spec_size = 2

    constraints = VOmegaConstraints(1.0, 0.8, 0.2, 0.3)
    model = ParallelTwoWheelVehicleModel([Circle(0.0, 0.0, 0.2)], dt, constraints)

    planner: MPPIPlanner = MPPIPlanner(model, stage_cost, terminal_cost)
    goal: jnp.ndarray = jnp.array([10.0, 10.0, math.pi / 2.0])
    fig, ax = plt.subplots()

    state: RobotState[VOmega] = RobotState(Point2D(0.5, 0.5, 0.0), VOmega(0.0, 0.0))

    # init SamplingData
    sampling_data: SamplingData = SamplingData(
        sampled_state_traj=jnp.zeros((num_samples, horizon, 5)),
        input_traj=jnp.zeros((horizon, act_spec_size)),
        input=jnp.zeros(act_spec_size),
        params=SamplingParams(
            num_samples=num_samples,
            horizon=horizon,
            lambda_=0.3,
            sigma_v=1.0,
            sigma_omega=0.8,
            act_spec_size=act_spec_size,
            key=jax.random.PRNGKey(0)
        )
    )

    # jit compile
    print("start jit compile")
    _ = planner.policy(jnp.array(state.to_numpy()), goal, sampling_data)
    print("end jit compile")

    start_time = time.perf_counter()
    states = []
    sampled_trajs_list = []

    for _ in range(200):
        act, sampling_data = planner.policy(jnp.array(state.to_numpy()), goal, sampling_data)
        state = model.step(state, act)
        print(state.to_numpy())
        states.append(state)
        sampled_trajs_list.append(sampling_data.sampled_state_traj)

    states = np.array(states)
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time - start_time} [s]")

    def update(state_sampled_trajs: jnp.ndarray, field):
        state, sampled_trajs = state_sampled_trajs
        ax.cla()
        ax.plot(goal[0], goal[1], "ro")
        ax.quiver(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]))
        ax.plot(0.5, 0.5, "go")
        ax.quiver(0.5, 0.5, np.cos(0.0), np.sin(0.0))

        field.frame.plot(ax, non_fill=True)
        ax.add_patch(plt.Circle((state.pos.x, state.pos.y), 0.2, fill=False))
        ax.quiver(state.pos.x, state.pos.y, jnp.cos(state.pos.theta), jnp.sin(state.pos.theta))
        for sampled_traj in sampled_trajs:
            ax.plot(sampled_traj[:, 0], sampled_traj[:, 1], "g", alpha=0.1)

    anim = FuncAnimation(fig, func=functools.partial(update, field=field),
                         frames=zip(states, sampled_trajs_list), interval=100, cache_frame_data=False)
    anim.save("output.gif", writer="pillow")
    plt.close()


if __name__ == "__main__":
    main()
