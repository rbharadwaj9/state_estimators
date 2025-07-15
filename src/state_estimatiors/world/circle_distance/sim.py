"""Simulation class for the world
Includes components such as the actual states and their evolution parameters, and the estimator.

TODO: Determine a formal interface that can be useful to perform analysis but that's basically developing a pseudo simulator for this task"""

class Simulation:

    def __init__(self) -> None:
        pass

    def step(self):
        """Evolve the state according to process noise"""
        pass

    def estimate(self):
        """Call estimator to estimate current state."""
        pass

    def reset(self):
        """Reset the simulation"""
        pass

    def run(self, T: int):
        """Run simulation loop for T timesteps"""
        # Maybe this can be a generator implementation to yield the metrics for analysis or collection?
