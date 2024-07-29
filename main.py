from models.robot_model import RobotState
from mppi import MPPIPlanner
from objects.field import Point2D, Field, GenTestField, Circle
import numpy as np
from models.veltypes import VOmega, VOmegaConstraints
from models.dynamics_model import ParallelTwoWheelVehicleModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import functools
import os
import time
import jax.numpy as jnp


def main():
    field: Field = GenTestField(0)
    dt = 0.1
    n_cpu: int = os.cpu_count()

    constraints = VOmegaConstraints(1.0, 0.8, 0.2, 0.3)
    model = ParallelTwoWheelVehicleModel([Circle(0.0, 0.0, 0.2)], dt, constraints)

    planner = MPPIPlanner(model, 40, 500, 0.3, 1.0, 0.8, n_cpu)
    planner.set_goal(jnp.array([10.0, 10.0, math.pi / 2.0]))
    fig, ax = plt.subplots()

    state: RobotState[VOmega] = RobotState(Point2D(0.5, 0.5, 0.0), VOmega(0.0, 0.0))

    # profile for planner.policy
    # pr = cProfile.Profile()
    # pr.enable()
    # act = planner.policy(state)
    # pr.disable()

    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    # jit compile
    print("start jit compile")
    planner.policy(state)
    print("end jit compile")

    start_time = time.perf_counter()
    states = []
    sampled_trajs_list = []

    for _ in range(200):
        act = planner.policy(state)
        state = model.step(state, act)
        print(state.to_numpy())
        states.append(state)
        sampled_trajs_list.append(planner.sampled_trajs)

    states = np.array(states)
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time - start_time} [s]")

    def update(state_sampled_trajs: jnp.ndarray, field, goal=planner.goal):
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
