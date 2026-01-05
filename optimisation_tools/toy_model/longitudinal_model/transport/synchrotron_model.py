import math

import optimisation_tools.toy_model.longitudinal_model.transport as transport
import transport.longitudinal_model

class SynchrotronModel(transport.longitudinal_model.LongitudinalModel):
    def __init__(self):
        super().__init__()
        self.phase_slip = -0.826 # particle revolution frequency df/f = - [phase_slip] dp/p
        self.momentum_min = 369.131 # momentum at ramp start
        self.momentum_max = self.momentum_min # momentum at ramp end
        self.magnet_period = 20e6 # ns time taken for the field to go through a complete oscillation (ramp up and down)
        self.magnet_minimum = 0.0 # ns time at which field is at minimum

    def get_magnet_momentum(self, t):
        momentum_swing = self.momentum_max-self.momentum_min
        magnet_phase = 2*math.pi*(t-self.magnet_minimum)/self.magnet_period-math.pi/2.0
        p_magnet = momentum_swing*(math.sin(magnet_phase)+1)/2+self.momentum_min
        return p_magnet

    def path_length(self, a_particle):
        """Return path length in mm"""
        # NOTE this should implement the momentum compaction factor to be correct
        path_length = self.r0*2*math.pi
        return path_length

    def get_time_of_flight(self, a_particle):
        p0 = self.get_magnet_momentum(a_particle.t)
        p1 = a_particle.momentum()
        dp = p1-p0
        f0 = a_particle.beta()*self.c_light/self.r0/2/math.pi
        df = -self.phase_slip*f0*dp/p0
        f1 = f0+df
        t1 = 1/f1
        return t1
