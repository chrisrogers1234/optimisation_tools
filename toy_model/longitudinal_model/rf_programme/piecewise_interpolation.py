import bisect
import math

import scipy.interpolate

import optimisation_tools.toy_model.longitudinal_model.rf_programme as rf_programme
import rf_programme.rf_programme

class QuickSpline:
    def __init__(self, x, y, k, s):
        self.x = x
        self.y = y
        if k != 1:
            raise NotImplementedError
        if s != 0:
            raise NotImplementedError

    def __call__(self, x_value):
        i = bisect.bisect_left(self.x, x_value)
        if i == len(self.x):
            i -= 1
        x0, x1 = self.x[i], self.x[i+1]
        y0, y1 = self.y[i], self.y[i+1]
        y_value = (x_value-x0)*(y1-y0)/(x1-x0)+y0
        return y_value


class PiecewiseInterpolation(rf_programme.rf_programme.RFProgramme):
    def __init__(self):
        self.v_list = []
        self.f_list = []
        self.t_list = []
        self.lookup_time = []
        self.k = 1 # fit order
        self.v_interpolation = None
        self.f_interpolation = None

    def const_frequency_setup(self, max_time):
        self.v_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.v_list, k=self.k, s=0)
        self.f_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.f_list, k=self.k, s=0)
        frequency = self.f_list[0]
        n = int(max_time*frequency)+1
        self.lookup_time = [i/frequency for i in range(n)]

    def setup(self, max_time):
        self.v_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.v_list, k=self.k, s=0)
        self.f_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.f_list, k=self.k, s=0)
        t = 0.0
        self.lookup_time = [t]
        while t < max_time: # phase is integral of 1/frequency...
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

