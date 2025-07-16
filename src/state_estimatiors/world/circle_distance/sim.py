"""Simulation class for the world
Includes components such as the actual states and their evolution parameters, and the estimator.

TODO: Determine a formal interface that can be useful to perform analysis but that's basically developing a pseudo simulator for this task"""

from dataclasses import MISSING, dataclass
from types import GeneratorType
import numpy as np

from state_estimatiors.bayesian_tracker import BayesianTracker
from state_estimatiors.world.circle_distance.dynamics_model import CircleDynamicsModel
from state_estimatiors.world.circle_distance.measurement_model import DistanceMeasurementModel
from state_estimatiors.world.model.dynamics_model import DynamicsModel
from state_estimatiors.world.model.measurement_model import MeasurementModel

@dataclass
class SimulationCfg:

    N: int = MISSING # Num states

    process_noise: float = MISSING

    sensor_noise: float = MISSING

    sensor_length: float = MISSING

class Simulation:

    def __init__(self, cfg: SimulationCfg) -> None:
        self.t_: int = 0
        self.cfg: SimulationCfg = cfg
        self.measurement_model: MeasurementModel = DistanceMeasurementModel(self.cfg.sensor_length, self.cfg.sensor_noise, self.cfg.N)
        self.dynamics_model: DynamicsModel = CircleDynamicsModel(self.cfg.process_noise, self.cfg.N)
        self.estimator: BayesianTracker = BayesianTracker(cfg.N, self.measurement_model, self.dynamics_model)
        self.reset()

    def _sample_process_noise(self):
        return np.random.default_rng().choice([-1, 1], size=1, p=[self.cfg.process_noise, 1-self.cfg.process_noise])

    def step(self):
        """Evolve the state according to process noise"""
        self.curr_state = (self.curr_state + self._sample_process_noise()) % self.cfg.N
        self.t_ += 1

    def estimate(self):
        """Call estimator to estimate current state."""
        raise NotImplementedError()

    def reset(self):
        """Reset the simulation"""
        self.curr_state = 0
        self.initialized = True

    def run(self, T: int):
        """Run simulation loop for T timesteps"""
        # Maybe this can be a generator implementation to yield the metrics for analysis or collection?
        raise NotImplementedError()

    def run_gen(self, T: int):
        """Run simulation loop for T timesteps"""
        for t in range(T):
            self.step()
            yield self.curr_state
