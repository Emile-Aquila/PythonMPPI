import numpy as np
from functools import total_ordering
import copy


@total_ordering
class Point2D:  # x, y, theta
    def __init__(self, x: float, y: float, theta: float = 0.0) -> None:
        self.x: float = x
        self.y: float = y
        self.theta: float = theta

    def __eq__(self, other: "Point2D") -> bool:
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __lt__(self, other: "Point2D") -> bool:
        if self.x == other.x:
            if self.y == other.y:
                return self.theta < other.theta
            else:
                return self.y < other.y
        else:
            return self.x < other.x

    def __add__(self, other: "Point2D") -> "Point2D":
        return Point2D(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other: "Point2D") -> "Point2D":
        return Point2D(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, other: float) -> "Point2D":
        return Point2D(self.x * other, self.y * other, self.theta)

    __rmul__ = __mul__

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.theta) + ")"

    def len(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def rotate(self, theta2: float) -> "Point2D":
        x = self.x * np.cos(theta2) - self.y * np.sin(theta2)
        y = self.x * np.sin(theta2) + self.y * np.cos(theta2)
        return Point2D(x, y, theta2 + self.theta)

    def unit(self) -> "Point2D":
        return copy.deepcopy(self) * (1.0 / self.len())

    def getXY(self) -> tuple[float, float]:
        return self.x, self.y

    def cross(self, other) -> "Point2D":  # 2次元での外積を求める.
        # \vec{a} \times \vec{b} = a_x b_y - a_y b_x
        return self.x * other.y - other.x * self.y

    def dot(self, other: "Point2D") -> float:
        return self.x * other.x + self.y * other.y

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
