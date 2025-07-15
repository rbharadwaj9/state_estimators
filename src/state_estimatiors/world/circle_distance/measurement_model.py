import numpy as np
from numpy.typing import NDArray

from state_estimatiors.world.circle_distance.utils import Utils
from state_estimatiors.world.model.measurement_model import MeasurementModel


class DistanceMeasurementModel(MeasurementModel):

    def __init__(self, L: float, e: float, N: int) -> None:
        super().__init__()
        self.L: float = L # Distance from center TODO: Chage to coordinate to allow more dimensions
        self.e: float = e # Sensor noise
        self.N: int = N

    def measure(self, z, x) -> NDArray[np.double]:
        theta = Utils.cvt_state_to_angle(x, self.N)
        dist = np.linalg.norm(np.array([self.L, 0]) - np.array([np.cos(theta), np.sin(theta)]))
        if np.abs(z - dist) <= self.e:
            return np.array(1/(2*self.e))
        return np.zeros(1)
