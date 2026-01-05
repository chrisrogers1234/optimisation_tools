import ctypes
import glob

import matplotlib
import scipy
import numpy

import optimisation_tools.toy_model.longitudinal_model.rf_programme
import rf_programme.piecewise_interpolation
import ROOT

class ExtractRFData:
    def __init__(self):
        self.file_glob = ""
        self.data = None
        self.time_key = "time"
        self.data_key = "data"
        self.stroke = 1 # take every nth data point
        self.time_units = 1e9
        self.time_step = -1 # autodetect
        self.volts_time_window = 2e3 # ns
        self.frequency_time_window = 2e3 # ns
        self.set_frequency = 1.352*1e-3 # GHz
        self.window_size = 100
        self.v_model = None
        self.output_directory = "./"
        self.problem_title = ""
        self.chi2 = -1.0

    def load_data(self):
        self.filename = glob.glob(self.file_glob)
        if len(self.filename) != 1:
            raise ValueError(f"File glob {self.file_glob} should define one file, found {self.filename}")
        self.data = numpy.load(self.filename[0])
        t_data = self.data[self.time_key]
        self.time_step =(t_data[-1]*self.time_units-t_data[0]*self.time_units)/(len(t_data)-1)
        print(f"Loaded RF file {self.filename[0]}: {len(self.data[self.data_key])} points with time step {self.time_step} ns")

    def inspect_data(self):
        for filename in sorted(self.data.keys()):
            print("In file", filename)
            print(type(self.data[filename]))
            print([key for key in self.data[filename].keys()])
            print([len(self.data[filename][key]) for key in self.data[filename].keys()])

    def specgram(self):
        v_data = self.data[self.data_key]
        t_data = self.data[self.time_key]
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.specgram(v_data, Fs = 1/self.time_step, NFFT=2**16)
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("Frequency [GHz]")
        axes.set_ylim(0, 1e-2)
        figure.savefig(f"{self.output_directory}/rf_specgram.png")
        matplotlib.pyplot.close(figure)

    def voltage_plot(self):
        v_data = self.data[self.data_key][::10]
        t_data = [t*self.time_units for t in self.data[self.time_key][::10]]
        v_fit = [self.v_model.get_voltage_magnitude(t) for t in self.t_peaks_list]
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.scatter(t_data, v_data, s=1)
        axes.plot(self.t_peaks_list, self.v_peaks_list, color="orange")
        axes.plot(self.t_peaks_list, v_fit, color="green")
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("Voltage [AU]")
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        figure.savefig(f"{self.output_directory}/voltage.png")
        matplotlib.pyplot.close(figure)

    def process_voltage_peaks(self):
        v_data = self.data[self.data_key]
        t_data = self.data[self.time_key]
        window = int(self.window_size/self.set_frequency/self.time_step)
        print(f"Processing voltage with window {window}")
        i0, i1 = -window, 0
        self.t_peaks_list = []
        self.v_peaks_list = []
        while i0+window < len(v_data):
            i0 += window
            i1 += window
            v_sublist = [abs(v) for v in v_data[i0:i1]]
            v_peak = max(v_sublist)
            #if len(self.v_peaks_list) and abs(v_peak - self.v_peaks_list[-1]) < 1e-6:
            #    continue
            v_index = v_sublist.index(v_peak)
            self.t_peaks_list.append(t_data[i0+v_index]*self.time_units)
            self.v_peaks_list.append(v_peak)
        self.t_peaks = numpy.array(self.t_peaks_list)
        self.v_peaks = numpy.array(self.v_peaks_list)

    def fit_voltage(self):
        global EXTRACT
        x_var = [1e6, 9e6]
        self.minuit = ROOT.TMinuit(4)
        self.minuit.DefineParameter(0, "t0", 1e6, 1e5, 0, 0)
        self.minuit.DefineParameter(1, "t1", 9e6, 1e5, 0, 0)
        self.minuit.DefineParameter(2, "v0", self.v_peaks_list[0], 0.1, 0, 0)
        self.minuit.DefineParameter(3, "v1", self.v_peaks_list[-1], 0.1, 0, 0)
        self.minuit.FixParameter(2)
        self.minuit.FixParameter(3)
        EXTRACT = self
        self.minuit.SetFCN(minuit_function)
        self.minuit.Command("SIMPLEX "+str(100)+" "+str(1e-3))
        self.linear_ramp_fit_function(1)

    def extract_data(self):
        self.process_voltage_peaks()
        self.fit_voltage()
        self.specgram()
        self.voltage_plot()

    def get_minuit_value(self, i):
        x = ctypes.c_double()
        err = ctypes.c_double()
        self.minuit.GetParameter(i, x, err)
        return x.value

    def linear_ramp_fit_function(self, verbose=0):
        # fit function like [flat, ramp, flat]
        self.chi2 = 0.0
        t0 = self.get_minuit_value(0)
        t1 = self.get_minuit_value(1)
        v0 = self.get_minuit_value(2)
        v1 = self.get_minuit_value(3)
        if t1 == t0:
            t0 -= 1
        if t1 < t0:
            t0, t1 = t1, t0
        self.v_model = rf_programme.piecewise_interpolation.PiecewiseInterpolation()
        self.v_model.t_list = [t0-1e9,  t0, t1, t1+1e9]
        self.v_model.v_list = [v0,    v0, v1, v1]
        self.v_model.f_list = [self.set_frequency]*len(self.v_model.t_list)
        self.v_model.const_frequency_setup(t1+1e9)
        v_estimated = numpy.array([self.v_model.get_voltage_magnitude(t) for t in self.t_peaks])
        delta_v = v_estimated-self.v_peaks
        self.chi2 = sum(delta_v**2)
        if verbose:
            print("    Trying fit with", t0, t1, v0, v1, "yields chi2", self.chi2)
        return self.chi2

EXTRACT = None

def minuit_function(nvar, parameters, score, jacobian, err):
    global EXTRACT
    pyscore = EXTRACT.linear_ramp_fit_function()
    score.value = pyscore


def main():
    extractor = ExtractRFData()
    extractor.file_glob = "data/CH5_111_*npz"
    extractor.load_data()
    extractor.extract_data()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")


