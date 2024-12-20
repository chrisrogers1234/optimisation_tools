import bisect
import math

import scipy.interpolate

import optimisation_tools.toy_model.longitudinal_model.rf_programme as rf_programme
import rf_programme.rf_programme


class PiecewiseInterpolation(rf_programme.rf_programme.RFProgramme):
    def __init__(self):
        self.v_list = []
        self.f_list = []
        self.t_list = []
        self.lookup_time = []
        self.k = 1

    def setup(self, max_time):
        self.v_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.v_list, k=self.k, s=0)
        self.f_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.f_list, k=self.k, s=0)
        t = 0.0
        self.lookup_time = [t]
        while t < max_time:
            t = self.lookup_time[-1]
            freq = float(self.f_interpolation(t))
            self.lookup_time.append(t+1./freq)

    def get_voltage_magnitude(self, t):
        return self.v_interpolation(t)

    def get_relative_time(self, t):
        index = bisect.bisect_left(self.lookup_time, t)
        try:
            dt = t-self.lookup_time[index]
        except IndexError:
            dt = t-self.lookup_time[-1]
        return dt

    def get_voltage(self, t):
        dt = self.get_relative_time(t)
        freq = self.f_interpolation(t)
        v0 = self.v_interpolation(t)
        volts = v0*math.sin(2*math.pi*freq*dt)
        return volts

    def get_frequency(self, t):
        return self.f_interpolation(t)

