import numpy as np
from numpy.typing import NDArray


class Utils:

    @staticmethod
    def cvt_state_to_angle(x: NDArray | int, N: int) -> NDArray | int:
        """Given discretized state x out of N states, convert to the angle in radians"""
        return 2*np.pi*x/N
