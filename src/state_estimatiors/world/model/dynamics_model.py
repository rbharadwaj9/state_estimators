import numpy
from numpy.typing import NDArray


class DynamicsModel:
    """Base class for dynamics model"""

    def step(self) -> NDArray[numpy.float64]:
        return self.step_all()

    def step_all(self) -> NDArray[numpy.float64]:
        """Give p(i | j) for all i, j"""
        raise NotImplementedError("Provide implementaiton for step_all()")

    # TODO: Move i, j into StateLike types
    def step_to(self, i) -> NDArray[numpy.float64]:
        """Given a current state i, provide PDF of p(i | j) for all j in a vector"""
        raise NotImplementedError("Provide implementaiton for step()")

    def step_one(self, i, j) -> NDArray[numpy.float64]:
        """Given a current state i, j return p(i | j)"""
        raise NotImplementedError("Provide implementaiton for step()")

