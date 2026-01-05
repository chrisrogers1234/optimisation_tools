import os
import numpy
import glob

class ExtractRealData:
    def __init__(self):
        self.file_glob = ""
        self.data = None
        self.time_key = "time"
        self.data_key = "data"
        self.stroke = 10 # take every nth data point
        self.time_units = 1
        self.t_hist = []
        self.t_bins = []
        self.t_resolution = 1

    def load_data(self):
        self.filename = glob.glob(self.file_glob)
        if len(self.filename) != 1:
            raise ValueError(f"File glob {self.file_glob} should define one file, found {self.filename}")
        self.data = numpy.load(self.filename[0])
        print("Loaded", self.filename[0])

    def inspect_data(self):
        for filename in sorted(self.data.keys()):
            print("In file", filename)
            print(type(self.data[filename]))
            print([key for key in self.data[filename].keys()])
            print([len(self.data[filename][key]) for key in self.data[filename].keys()])

    def extract_data(self):
        self.t_hist = self.data[self.data_key][::self.stroke]
        self.t_bins = self.data[self.time_key][::self.stroke]*self.time_units
        my_t_bins = numpy.array([self.t_bins[:-1], self.t_bins[1:]])
        my_t_bins = my_t_bins.transpose()
        self.t_bins = numpy.array([(t[0]+t[1])/2 for t in my_t_bins])
        self.t_bins = numpy.insert(self.t_bins, 0, 2*self.t_bins[0]-self.t_bins[1])
        self.t_bins = numpy.insert(self.t_bins, -1, 2*self.t_bins[-1]-self.t_bins[-2])
        self.t_resolution = abs(self.t_bins[-1]-self.t_bins[0])/(len(self.t_bins)-1)
        print("T bins", len(self.t_bins), self.t_bins[:5], self.t_bins[-5:])
        print("T hist", len(self.t_hist), self.t_hist[:5], self.t_hist[-5:])
        print("Bin width", self.t_resolution)

if __name__ == "__main__":
    import optimisation_tools.toy_model.longitudinal_model.analysis.monitor_analysis as monitor_analysis
    extractor = ExtractRealData()
    extractor.file_glob = "data/CH1*59*npz"
    extractor.load_data()
    extractor.extract_data()

    analysis = monitor_analysis.MonitorAnalysis()

