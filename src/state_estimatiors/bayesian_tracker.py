import numpy as np
from numpy.typing import NDArray

from state_estimatiors.estimator import Estimator
from state_estimatiors.world.model.dynamics_model import DynamicsModel
from state_estimatiors.world.model.measurement_model import MeasurementModel


class BayesianTracker(Estimator):
    """Discrete State Bayesian Tracker"""

    def __init__(self, N: int, measure_model: MeasurementModel, dynamics_model: DynamicsModel) -> None:
        super().__init__(measure_model, dynamics_model)
        self.N: int = N # num states (implicitly assume each state is also an int between 1 and N to simplify)

        self.a_k_k: NDArray[np.double] = np.zeros(self.N) # Posterior
        self.a_k_k_1: NDArray[np.double] = np.zeros(self.N) # Prior

        self.reset()

    def reset(self, init_distribution: NDArray[np.double] | None = None) -> None:
        """Initialize the estimator by setting default values"""
        super().reset()
        self.a_k_k = init_distribution if init_distribution is not None else np.ones((self.N, 1)) / self.N # Uniform distribution

    def estimate(self, z_bar: NDArray[np.double]):
        """Perform estimation for one step for a particular measurement"""
        if self.t_ is 0:
            raise ValueError("Please increment the timer")
        self._prior_update()
        self._measurement_update(z_bar)

    def _prior_update(self) -> None:
        # Perform update, and expand to a_k_k_1
        if self.t_ <= self.a_k_k.shape[1]:
            raise ValueError("Plese increment time first")

        a_new: NDArray[np.double] = np.zeros((self.N, 1))
        # TODO: Vectorize
        for i in range(self.N):
            p_i_for_all_j: NDArray[np.double] = self.dynamics_model_.step_to(i)
            a_new[i] = np.sum(p_i_for_all_j @ self.a_k_k[self.t - 1, :])

        self.a_k_k_1 = np.hstack((self.a_k_k_1, a_new))

    def _measurement_update(self, z_bar: NDArray[np.double]) -> None:
        """
        z_bar is the given measurement for which we would like to update our estimate
        """
        # Perform update, and expand to a_k_k
        if self.t_ <= self.a_k_k.shape[1]:
            raise ValueError("Plese increment time first")

        norm: float = 0.0
        # TODO: Vectorize
        for j in range(self.N):
            norm += self.measure_model_.measure(z_bar, j) * self.a_k_k_1[j, self.t]
        # TODO: Vectorize
        a_new: NDArray[np.double] = np.zeros((self.N, 1))
        for i in range(self.N):
            a_new[i] = (self.measure_model_.measure(z_bar, i) * self.a_k_k_1[i, self.t]) / norm

        self.a_k_k = np.hstack((self.a_k_k, a_new))

    @property
    def est_state(self) -> NDArray[np.double]:
        if self.t_ > self.a_k_k.shape[1]:
            raise ValueError("Time variable exceeds estimations. Please ensure estimation is done?")
        return self.a_k_k[:, self.t_]
