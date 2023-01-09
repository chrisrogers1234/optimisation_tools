import os
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
        print(f"  Loaded {line_number} lines")

    def filter_data(self):
        self.v_fft = numpy.fft.fft(self.v_array)
        # lowest frequency corresponds to 1/dt/n 
        # highest frequency corresponds to 1/dt
        v_fft_cut = copy.deepcopy(self.v_fft)
        if self.low_frequency_cutoff:
            low_frequency_index = int(self.low_frequency_cutoff/self.frequency_bin)
            v_fft_cut[:low_frequency_index] = numpy.zeros((low_frequency_index,))
            print(f"  Low frequency cut at index {low_frequency_index} for voltage data from {self.t_array[0]} to {self.t_array[-1]} ns")
        self.v_output = numpy.fft.ifft(v_fft_cut)

    def write_data(self):
        fout = open(os.path.join(self.output_directory, "filtered.dat"), "w")
        for i, t in enumerate(self.t_array):
            v = self.v_output[i]
            fout.write(str(t)+" "+str(numpy.real(v))+"\n")

    def plot_data(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        time_constant = self.t_array[1]-self.t_array[0]
        axes.bar(self.t_array, self.v_array, width=time_constant, align="edge")
        axes.bar(self.t_array, self.v_output, width=time_constant)
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Voltage [AU]")
        figure.savefig(self.output_directory+"/monitor_and_filtered.png")
        t_min, t_max = 0, 2e4
        i = 0
        while False and t_min < self.t_array[-1]:
            axes.set_xlim(t_min, t_max)
            suffix = str(i).rjust(4, "0")
            t_min += 5e3
            t_max += 5e3
            i += 1
            figure.savefig(self.output_directory+"/monitor_and_filtered_"+suffix+".png")
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        time_constant = self.t_array[1]-self.t_array[0]
        axes.bar(self.t_array, self.v_output, width=time_constant)
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Voltage [AU]")
        figure.savefig(self.output_directory+"/filtered.png")


def main_dir(output_directory):
    print("Doing", output_directory)
    model = MonitorModel()
    model.output_directory = output_directory
    print("  Loading")
    model.load_data(output_directory+"/monitor.dat")
    print("  Filtering")
    model.filter_data()
    print("  Plotting")
    model.plot_data()
    print("  Writing")
    model.write_data()

def main():
    output_directory_list = [
        f"output/kurns_v4/constant_bucket_beam_energy_scan_{i}/" for i in range(-10, 11, 2)
    ]
    print("Loading following dirs")
    for output_directory in output_directory_list:
        print("   ", output_directory)
    for output_directory in output_directory_list:
        a_pid = os.fork()
        if a_pid == 0: # the child process
            main_dir(output_directory)
            # hard exit returning 0 - don't want to end up in any exit handling
            # stuff, just die ungracefully now the simulation has run
            os._exit(0)
        else:
            retvalue = os.waitpid(a_pid, 0)[1]
        if retvalue != 0:
            # it means we never reached os._exit(0)
            raise RuntimeError("Opal failed returning "+str(retvalue))
        print("PARENT - next")

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")