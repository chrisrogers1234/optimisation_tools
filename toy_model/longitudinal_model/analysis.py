import csv
import os
import json
import copy
import numpy
import matplotlib.pyplot

class Analysis(object):
    def __init__(self, output_directory):
        self.t_array = []
        self.v_filtered = []
        self.rf_array = []
        self.parameters = {}
        self.output_directory = output_directory
        self.monitor_file_name = os.path.join(self.output_directory, "filtered.dat")
        self.config_file_name = os.path.join(self.output_directory, "config.json")
        self.rf_file_name = os.path.join(self.output_directory, "rf.dat")

    def load_config(self):
        with open(self.config_file_name) as fin:
            self.parameters = json.loads(fin.read())
            self.rf_frequency = self.parameters["rf_frequency"]
        print("Loaded parameters:", json.dumps(self.parameters, indent=2))

    def load_filtered(self):
        self.t_array = []
        self.v_array = []
        line_number = 0
        with open(self.monitor_file_name) as fin:
            for line in fin.readlines():
                line_number += 1
                words = line.split()
                if len(words) < 2:
                    print(f"Closing on line {line_number}: {line}")
                    break
                self.t_array.append(float(words[0]))
                self.v_array.append(float(words[1]))
        self.t_array = numpy.array(self.t_array)
        self.v_array = numpy.array(self.v_array)

    def load_rf(self):
        with open(self.rf_file_name) as fin:
            csv_reader = csv.reader(fin, delimiter=' ')
            self.rf_data = numpy.array([row for row in csv_reader])
        self.rf_data = self.rf_data.transpose().astype(float)
        self.time_step = self.rf_data[0][1]-self.rf_data[0][0] # ns
        print("rf frequency", self.rf_frequency)

    def load_data(self):
        self.load_filtered()
        self.load_config()
        self.load_rf()

    def smooth(self):
        pass

    def windowed_analysis(self, window_size, window_advance):
        print("Windowed analysis with time step", window_size*self.time_step)
        i0, i1 = 0, window_size
        n_points = len(self.v_array)
        self.delta_volts_t_array = []
        self.delta_volts_array = []
        while i1 < n_points:
            v_max = numpy.amax(self.v_array[i0:i1])
            v_min = numpy.amin(self.v_array[i0:i1])
            self.delta_volts_array.append(v_max-v_min)
            self.delta_volts_t_array.append((i1+i0)*self.time_step/2)
            i0 += window_advance
            i1 += window_advance
        out_str = json.dumps({"t_list":self.delta_volts_t_array, "v_p2p_list":self.delta_volts_array})
        fout = open(os.path.join(self.output_directory, "analysis.dat"), "w")
        fout.write(out_str)

def main():
    output_directory_list = [
#        f"output/kurns_v4/constant_bucket_beam_energy_scan_{i}/" for i in range(-10, 11, 2)
        "output/kurns_v4/constant_bucket_v1/"
    ]
    for output_directory in output_directory_list:
        model = Analysis(output_directory)
        print("Loading")
        model.load_data()
        model.smooth()
        rf_period = 1./(model.rf_frequency*model.time_step) # rf period in time steps
        model.windowed_analysis(int(rf_period*2.0), int(rf_period/2))

if __name__ == "__main__":
    main()
