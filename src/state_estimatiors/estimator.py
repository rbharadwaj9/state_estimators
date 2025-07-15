"""
Base State Estimator Class

Each estimator has a measurement model and dynamics model
"""
from numpy.typing import NDArray

from state_estimatiors.world.model.dynamics_model import DynamicsModel
from state_estimatiors.world.model.measurement_model import MeasurementModel


class Estimator:

    def __init__(self, measure_model: MeasurementModel, dynamics_model: DynamicsModel) -> None:
        self.measure_model_: MeasurementModel = measure_model
        self.dynamics_model_: DynamicsModel = dynamics_model

        self.t_: int = 0

    def reset(self) -> None:
        """Initialize the estimator by setting default values"""
        self.t_ = 0

    def estimate(self, z_bar: NDArray[np.double]):
        """Estimate for a given measurement"""
        raise NotImplementedError()

    def increment_t(self) -> None:
        self.t_ += 1

    @property
    def t(self) -> int:
        return self.t_
