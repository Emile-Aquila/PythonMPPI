import dataclasses
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from robot_model import Constraints


def clip(value: float, min_value: float, max_value: float) -> float:
    return max(min(max_value - 1e-3, value), min_value + 1e-3)


@dataclasses.dataclass(frozen=True)
class VOmega:
    v: float  # 符号付きのvelの大きさ
    omega: float  # \dot{theta}
    size: int = 2

    def __add__(self, other):
        return self.__class__(self.v + other.v, self.omega + other.omega)

    def __mul__(self, other: float):
        return self.__class__(self.v * other, self.omega * other)

    __rmul__ = __mul__

    def __truediv__(self, other: float):
        return self.__class__(self.v / other, self.omega / other)

    def __sub__(self, other):
        return self.__class__(self.v - other.v, self.omega - other.omega)

    def __abs__(self):
        return self.__class__(abs(self.v), abs(self.omega))

    def weighted_sum(self, weights: list[float]) -> float:
        return self.v * weights[0] + self.omega * weights[1]

    def to_numpy(self) -> np.ndarray:
        return np.array([self.v, self.omega])


class VOmegaConstraints(Constraints[VOmega]):
    max_v: float
    max_omega: float

    max_d_v: float
    max_d_omega: float

    dt: float

    def __init__(self, max_v: float, max_omega: float, max_d_v: float, max_d_omega: float) -> None:
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_d_v = max_d_v
        self.max_d_omega = max_d_omega

    def get_feasible_input_space(self, vel: VOmega | np.ndarray) -> np.ndarray:
        if isinstance(vel, VOmega):
            max_v = min(self.max_v, vel.v + self.max_d_v)
            min_v = max(-self.max_v, vel.v - self.max_d_v)

            max_omega = min(self.max_omega, vel.omega + self.max_d_omega)
            min_omega = max(-self.max_omega, vel.omega - self.max_d_omega)
        else:
            max_v = np.minimum(self.max_v, vel[0] + self.max_d_v)
            min_v = np.maximum(-self.max_v, vel[0] - self.max_d_v)

            max_omega = np.minimum(self.max_omega, vel[1] + self.max_d_omega)
            min_omega = np.maximum(-self.max_omega, vel[1] - self.max_d_omega)

        return np.array([[min_v, max_v], [min_omega, max_omega]])

    def clip_act(self, act_pre: VOmega, act: VOmega) -> VOmega:
        input_spec: np.ndarray = self.get_feasible_input_space(act_pre)
        v = clip(act.v, input_spec[0][0], input_spec[0][1])
        omega = clip(act.omega, input_spec[1][0], input_spec[1][1])
        return VOmega(v, omega)

    def clip_act_numpy(self, act_pre: np.ndarray, act: np.ndarray) -> np.ndarray:
        input_spec: np.ndarray = self.get_feasible_input_space(act_pre)
        v = clip(act[0], input_spec[0][0], input_spec[0][1])
        omega = clip(act[1], input_spec[1][0], input_spec[1][1])
        return np.array([v, omega])
