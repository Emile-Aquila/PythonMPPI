import math
from objects.field import Point2D, Object
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from Robot_model import RobotState, KinematicsModel
from models.veltypes import VOmega, VOmegaConstraints


class ParallelTwoWheelVehicleModel(KinematicsModel[VOmega]):
    def __init__(self, objects: list[Object], dt: float, constraints: VOmegaConstraints):
        super(ParallelTwoWheelVehicleModel, self).__init__(objects, dt, constraints)

    @staticmethod
    def kinematics(state: RobotState[VOmega], act: VOmega, dt: float) -> RobotState[VOmega]:
        vel = (act + state.vel) * 0.5
        new_pos = state.pos + Point2D(vel.v * dt * math.cos(state.pos.theta),
                                      vel.v * dt * math.sin(state.pos.theta),
                                      vel.omega * dt)
        return RobotState[VOmega](new_pos, act)

    @staticmethod
    def kinematics_numpy(state: np.ndarray, act: np.ndarray, dt: float) -> np.ndarray:
        vel = (act + state[3:]) * 0.5
        new_pos = state[:3] + np.array([vel[0] * dt * math.cos(state[2]),
                                        vel[0] * dt * math.sin(state[2]),
                                        vel[1] * dt])
        new_pos = np.concatenate([new_pos, act])
        return new_pos
