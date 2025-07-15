from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

MeasurmentLike = TypeVar('Z')
StateLike = TypeVar('X')

"""Base Class for a Measurement Model"""
class MeasurementModel:
    def measure(self, z, x) -> NDArray[np.double]:
        """Returns p(z | x)"""
        raise NotImplementedError("Provide implementation for measure()")

