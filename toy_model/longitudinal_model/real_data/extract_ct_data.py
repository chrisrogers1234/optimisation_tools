import bisect
import ctypes
import glob

import matplotlib
import scipy
import numpy

class ExtractCTData:
    def __init__(self):
        self.file_glob = ""
        self.data = None
        self.time_key = "time"
        self.data_key = "data"
        self.stroke = 1 # take every nth data point
        self.time_units = 1e9
        self.time_step = -1 # autodetect
        self.output_directory = "./"
        self.v_max_index = 0
        self.fit_tolerance = 1e-8
        self.fit_order = 3
        self.fit_function = None
        self.stroke1 = 1000
        self.stroke2 = 10
        self.offset = 0.0
        self.offset_time = 100000/self.time_units # 1e5 ns before injection starts
        self.problem_title = ""

    def load_data(self):
        self.filename = glob.glob(self.file_glob)
        if len(self.filename) != 1:
            raise ValueError(f"File glob {self.file_glob} should define one file, found {self.filename}")
        self.data = numpy.load(self.filename[0])
        t_data = self.data[self.time_key]
        self.time_step =(t_data[-1]*self.time_units-t_data[0]*self.time_units)/(len(t_data)-1)
        self.v_max_index = self.data[self.data_key].argmax()
        print(f"Loaded CT file {self.filename[0]}: {len(self.data[self.data_key])} points with time step {self.time_step} ns")

    def fit_data(self):
        print("Making smoothing spline...", end="")
        self.fit_function = scipy.interpolate.make_smoothing_spline(
                self.data[self.time_key][::self.stroke1], 
                self.data[self.data_key][::self.stroke1], lam=1e-9)
        self.get_offset()
        print(" ... done")

    def get_offset(self):
        offset_index = bisect.bisect_left(self.data[self.time_key], self.offset_time)
        print("Offset index", offset_index)
        mean_v = numpy.mean(self.data[self.data_key][:offset_index])
        self.offset = mean_v

    def voltage_plot(self):
        v_data = self.data[self.data_key][::self.stroke2 ]
        t_data = self.data[self.time_key][::self.stroke2 ]*self.time_units
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.scatter(t_data, v_data, s=1)

        t_stroked = self.data[self.time_key][::self.stroke2 ]*self.time_units
        fit_data = [self.fitted_not_offset(t) for t in t_stroked]
        axes.plot(t_stroked, fit_data, color="orange")

        t_stroked = self.data[self.time_key][::self.stroke2 ]*self.time_units
        fit_data = [self.fitted(t) for t in t_stroked]
        axes.plot(t_stroked, fit_data, color="green")

        axes.set_xlabel("time [ns]")
        axes.set_ylabel("Voltage [AU]")
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        figure.savefig(f"{self.output_directory}/real_ct.png")
        matplotlib.pyplot.close(figure)

    def fitted(self, t):
        return self.fit_function(t/self.time_units)-self.offset

    def fitted_not_offset(self, t):
        return self.fit_function(t/self.time_units)

    def extract_data(self):
        self.fit_data()
        self.voltage_plot()


