from . import constants

class Particle(object):
    """
    Describes a particle; holds mass, momentum, energy, charge etc info
    """
    def __init__(self):
        """
        Initialise the particle to "NULL"
        """
        self.p = 0. # total momentum
        self.mass = 0.
        self.charge = 0.
        self.energy = 0. # total energy
        self.gamma_rel = 1.
        self.beta_rel = 0.
        self.pid = 0 # PDG PID code

    @classmethod
    def new_from_ke(cls, kinetic_energy, pid):
        """
        Generate a new particle based on kinetic energy
          - kinetic_energy: KE in MeV
          - pid: pdg pid number
        Returns an object of type particle
        """
        particle = Particle()
        particle.set_pid(pid)
        particle.set_kinetic_energy(kinetic_energy)
        return particle

    @classmethod
    def new_from_momentum(cls, momentum, pid):
        """
        Generate a new particle based on momentum
          - momentum: KE in MeV
          - pid: pdg pid number
        Returns an object of type particle
        """
        particle = Particle()
        particle.set_pid(pid)
        particle.set_momentum(momentum)
        return particle

    def set_pid(self, new_pid):
        """
        Change the pid; preserves energy and momentum, even if this leaves the
        total energy < mass which is non-physical
        """
        self.mass = constants.get_mass(new_pid)
        self.charge = constants.get_charge(new_pid)
        self.pid = new_pid

    def get_momentum(self):
        return self.p

    def set_momentum(self, new_momentum):
        self.p = new_momentum
        self.energy = (self.p**2+self.mass**2)**0.5
        self.gamma_rel = self.energy/self.mass
        self.beta_rel = self.p/self.energy

    def get_kinetic_energy(self):
        return self.energy - self.mass

    def set_kinetic_energy(self, new_ke):
        self.energy = new_ke + self.mass
        self.gamma_rel = self.energy/self.mass
        self.p = (self.energy**2-self.mass**2)**0.5
        self.beta_rel = self.p/self.energy

    def set_energy(self, new_energy):
        self.energy = new_energy
        self.gamma_rel = self.energy/self.mass
        self.p = (self.energy**2-self.mass**2)**0.5
        self.beta_rel = self.p/self.energy
