import copy
import numpy.fft
import matplotlib.pyplot

class MonitorModel(object):
    def __init__(self):
        self.t_array = []
        self.v_array = []
        self.v_fft = []
        self.v_output = []
        self.frequency_bin = 1
        self.low_frequency_cutoff = 1e-4 # [GHz]
        self.output_directory = None

    def load_data(self, monitor_file_name):
        self.t_array = []
        self.v_array = []
        line_number = 0
        with open(monitor_file_name) as fin:
            for line in fin.readlines():
                line_number += 1
                words = line.split()
                if len(words) < 2:
                    print(f"Closing on line {line_number}: '{line}'")
                    break
                self.t_array.append(float(words[0]))
                self.v_array.append(float(words[1]))
        self.t_array = numpy.array(self.t_array)
        self.v_array = numpy.array(self.v_array)
        self.frequency_bin = 1/(self.t_array[-1]-self.t_array[0])
        print(f"Loaded {line_number} lines")

    def filter_data(self):
        self.v_fft = numpy.fft.fft(self.v_array)
        # lowest frequency corresponds to 1/dt/n 
        # highest frequency corresponds to 1/dt
        v_fft_cut = copy.deepcopy(self.v_fft)
        if self.low_frequency_cutoff:
            low_frequency_index = int(self.low_frequency_cutoff/self.frequency_bin)
            v_fft_cut[:low_frequency_index] = numpy.zeros((low_frequency_index,))
            print(f"Low frequency cut at index {low_frequency_index} for voltage data from {self.t_array[0]} to {self.t_array[-1]} ns")
        self.v_output = numpy.fft.ifft(v_fft_cut)

    def write_data(self):
        pass

    def plot_data(self):
        figure = matplotlib.pyplot.figure()
        print("monitor and filtered")
        axes = figure.add_subplot()
        time_constant = self.t_array[1]-self.t_array[0]
        axes.bar(self.t_array, self.v_array, width=time_constant)
        axes.bar(self.t_array, self.v_output, width=time_constant)
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Voltage [AU]")
        figure.savefig(self.output_directory+"/monitor_and_filtered.png")
        t_min, t_max = 0, 2e4
        i = 0
        while True and t_min < self.t_array[-1]:
            print("Subplot", i)
            axes.set_xlim(t_min, t_max)
            suffix = str(i).rjust(4, "0")
            t_min += 5e3
            t_max += 5e3
            i += 1
            figure.savefig(self.output_directory+"/monitor_and_filtered_"+suffix+".png")
        print("filtered")
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        time_constant = self.t_array[1]-self.t_array[0]
        axes.bar(self.t_array, self.v_output, width=time_constant)
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Voltage [AU]")
        figure.savefig(self.output_directory+"/filtered.png")
        return
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        f_array = [i*self.frequency_bin for i in range(len(self.v_fft)-1)]
        axes.bar(f_array, self.v_fft[1:], width=self.frequency_bin)
        axes.set_xlabel("Frequency [GHz]")
        axes.set_ylabel("FFT(V) [AU]")
        figure.savefig(self.output_directory+"/fft_voltage.png")


def main():
    output_directory = "output/hold_ramp_hold_stats_500_50_500/"
    model = MonitorModel()
    model.output_directory = output_directory
    print("Loading")
    model.load_data(output_directory+"/monitor.dat")
    print("Filtering")
    model.filter_data()
    print("Plotting")
    model.plot_data()
    print("Writing")
    model.write_data()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")