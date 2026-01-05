
import optimisation_tools.toy_model.longitudinal_model.beam as beam
import beam.beam_factory

class Particle(object):
    def __init__(self, t, energy, mass=None):
        self.t = t
        self.energy = energy
        self.mass = mass
        if self.mass == None:
            self.mass = beam.beam_factory.BeamFactory.mass

    def set_momentum(self, new_momentum):
        self.energy = (new_momentum**2+self.mass**2)**0.5-self.mass

    def momentum(self):
        total_energy = self.energy + self.mass
        momentum = (total_energy**2-self.mass**2)**0.5
        return momentum

    def beta(self):
        if self.energy <= 0.0:
            return 0.0
        total_energy = self.energy + self.mass
        momentum = (total_energy**2-self.mass**2)**0.5
        beta = momentum/total_energy
        return beta
