import dataclasses
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
import copy
import sys
import os
import numpy as np
import jax.numpy as jnp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from objects.field import Point2D, Object

VelType = TypeVar("VelType")


@dataclasses.dataclass(frozen=True)
class RobotState(Generic[VelType]):
    pos: Point2D
    vel: VelType

    def to_numpy(self) -> np.ndarray:
        return np.concatenate([self.pos.to_numpy(), self.vel.to_numpy()])


class RobotModelBase(ABC):
    _objects: list[Object]

    def __init__(self, objects: list[Object]):
        self._objects = objects

    @staticmethod
    def _generate_new_obj(obj: Object, pos: Point2D) -> Object:
        ans = copy.deepcopy(obj)
        ans.change_pos(pos + ans.pos)
        return ans

    def get_objects(self, state: RobotState) -> list[Object]:
        return list(map(lambda x: self._generate_new_obj(x, state.pos), self._objects))

    def plot(self, ax):
        for tmp in self._objects:
            tmp.plot(ax)
        ax.set_aspect("equal")


ActType = TypeVar("ActType")


class Constraints(Generic[ActType], ABC):
    @abstractmethod
    def get_feasible_input_space(self, act: ActType | np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def clip_act(self, act_pre: ActType, act: ActType) -> ActType:
        raise NotImplementedError()

    @abstractmethod
    def clip_act_numpy(self, act_pre: np.ndarray, act: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def clip_act_jax(self, act_pre: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()


class RobotModel(Generic[ActType], RobotModelBase):
    def __init__(self, objects: list[Object]):
        super(RobotModel, self).__init__(objects)

    @abstractmethod
    def step(self, state: RobotState, act: ActType) -> RobotState:
        raise NotImplementedError()


class KinematicsModel(Generic[ActType], RobotModel[ActType]):
    def __init__(self, objects: list[Object], dt: float, constraints: Constraints[ActType]):
        super(KinematicsModel, self).__init__(objects)
        self.dt: float = dt
        self.constraints: Constraints[ActType] = constraints

    def step(self, state: RobotState, act: ActType) -> RobotState:
        return self.kinematics(state, act, self.dt)

    @staticmethod
    def kinematics(state: RobotState[ActType], act: ActType, dt: float) -> RobotState[ActType]:
        raise NotImplementedError()

    @staticmethod
    def kinematics_numpy(self, state: np.ndarray, act: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
