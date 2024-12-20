import math

import optimisation_tools.toy_model.longitudinal_model.transport as transport
import transport.longitudinal_model

class SynchrotronModel(transport.longitudinal_model.LongitudinalModel):
    def __init__(self):
        super().__init__()
        self.phase_slip = -0.826
        self.momentum_min = 369.131 # 70 MeV
        self.momentum_max = self.momentum_min
        self.magnet_period = 20e6 # ns
        self.t0 = 0.0 # point at which field is at minimum
        #1463.2960055983203

    def get_magnet_momentum(self, t):
        momentum_swing = self.momentum_max-self.momentum_min
        magnet_phase = 2*math.pi*(t-self.t0)/self.magnet_period-math.pi/2.0
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
        t0 = self.r0*2*math.pi/a_particle.beta()/self.c_light
        dt = -t0*p0/dp/self.phase_slip # (linear) phase slip is domega/dp  * p/omega
        t1 = t0+dt
        return t1
