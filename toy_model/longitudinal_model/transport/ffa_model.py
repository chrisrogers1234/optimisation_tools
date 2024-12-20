import math

import optimisation_tools.toy_model.longitudinal_model.transport as transport
import transport.longitudinal_model

class FFAModel(transport.longitudinal_model.LongitudinalModel):
    def __init__(self):
        super().__init__()
        self.k = 1 # gradient

    def path_length(self, a_particle):
        """Return path length in mm"""
        p0 = ((self.injection_momentum+self.mass)**2-self.mass**2)**0.5 # injection momentum
        p1 = a_particle.momentum()
        r1 = self.r0*(p1/p0)**(1/(self.k+1)) # k+1 from b r = p q; note radial dependence
        return r1*2*math.pi

    def get_time_of_flight(self, a_particle):
        beta = a_particle.beta()
        dt = self.path_length(a_particle)/beta/self.c_light
        return dt