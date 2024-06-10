from models.Robot_model import RobotState
from mppi import MPPIPlanner
from objects.field import Point2D, Field, GenTestField, Circle
import numpy as np
from models.veltypes import VOmega, VOmegaConstraints
from models.DynamicsModel import ParallelTwoWheelVehicleModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import functools


def main():
    field: Field = GenTestField(0)
    dt = 0.1

    constraints = VOmegaConstraints(1.0, 0.8, 0.2, 0.3)
    model = ParallelTwoWheelVehicleModel([Circle(0.0, 0.0, 0.2)], dt, constraints)

    planner = MPPIPlanner(model, 40, 500, 0.3, 1.0, 0.8)
    planner.set_goal(np.array([10.0, 10.0, math.pi / 2.0]))
    fig, ax = plt.subplots()

    state: RobotState[VOmega] = RobotState(Point2D(0.5, 0.5, 0.0), VOmega(0.0, 0.0))
    print(state.to_numpy())

    states = []
    sampled_trajs_list = []
    for _ in range(200):
        act = planner.policy(state)
        state = model.step(state, act)
        print(state.to_numpy())
        states.append(state)
        sampled_trajs_list.append(planner.sampled_trajs)

    states = np.array(states)

    def update(state_sampled_trajs, field, model, goal=planner.goal):
        state, sampled_trajs = state_sampled_trajs
        ax.cla()
        ax.plot(goal[0], goal[1], "ro")
        ax.quiver(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]))
        ax.plot(0, 0, "go")
        field.frame.plot(ax, non_fill=True)
        model.plot(ax)
        ax.quiver(state.pos.x, state.pos.y, np.cos(state.pos.theta), np.sin(state.pos.theta))
        for sampled_traj in sampled_trajs:
            ax.plot(sampled_traj[:, 0], sampled_traj[:, 1], "g", alpha=0.1)

    anim = FuncAnimation(fig, func=functools.partial(update, field=field, model=model),
                         frames=zip(states, sampled_trajs_list), interval=100)
    anim.save("output.mp4", writer="ffmpeg")
    plt.close()


if __name__ == "__main__":
    main()
