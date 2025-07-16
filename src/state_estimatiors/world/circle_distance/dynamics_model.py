import numpy as np
from numpy._typing import _ShapeLike
from numpy.typing import NDArray

from state_estimatiors.world.model.dynamics_model import DynamicsModel


class CircleDynamicsModel(DynamicsModel):

    def __init__(self, r: float, N: int) -> None:
        super().__init__()
        self.r: float = r
        self.N: int = N

    def _process_noise(self, size: _ShapeLike = 1) -> NDArray[np.double]:
        """Sample process noise"""
        return np.random.default_rng().choice([1, -1], size=size, p=self.r)

    def step_to(self, i) -> NDArray[np.float64]:
        rval = np.zeros(self.N)
        # TODO: Vectorize
        for j in range(self.N):
            rval[j] = self.step_one(i, j)
        return rval

    def step_one(self, i, j) -> NDArray[np.float64]:
        if i == (j + 1) % self.N:
            return np.array(self.r)
        elif i == (j - 1) % self.N:
            return np.array(1 - self.r)
        else:
            return np.array(0)
