import os
import glob
import json

import numpy
import matplotlib
import matplotlib.pyplot


class PlotFlukaYield:
    def __init__(self):
        self.plot_dir = ""
        self.source_file_glob = ""
        self.plot_data = []
        self.data_sum = None
        self.proton_energy = 0.0

    def load_files(self):
        file_list = glob.glob(self.file_glob)
        if len(file_list) == 0:
            print("Could not find any files from ", self.file_glob)
        for a_file in file_list:
            self.parse_one_file(a_file)
        self.sum_data()

    def parse_n_p(self, line, data):
        line = line.split("followed")[1]
        n_p = line.split(",")[0]
        n_p = float(n_p)
        data["n_protons"] = n_p

    def parse_bin(self, line, type, data):
        if "linear" not in line:
            raise RuntimeError("Not linear binning", line)
        if type not in line:
            raise RuntimeError("Type was not", type, line)
        line = line.replace(",", " ")
        words = line.split()
        low = float(words[4])
        high = float(words[6])
        n_bins = float(words[8])
        data[type] = {
            "low":low,
            "high":high,
            "n_bins":n_bins,
            "bins":numpy.linspace(low, high, int(n_bins+1)).tolist(),
            "bin_width":(high-low)/n_bins
        }

    def sum_data(self):
        self.data_sum = [d for d in self.plot_data[0]["bin_contents"]]
        for item in self.plot_data[1:]:
            if len(item["bin_contents"]) != len(self.data_sum):
                raise ValueError("Bin length mismatched")
            for i, d in enumerate(item["bin_contents"]):
                self.data_sum[i] += d

    def parse_one_file(self, a_file):
        data = {}
        fin = open(a_file)
        for i in range(6):
            line = fin.readline()
        self.parse_n_p(line, data)
        for i in range(8):
            line = fin.readline()
        self.parse_bin(line, "energy", data)
        line = fin.readline()
        self.parse_bin(line, "angular", data)
        line = fin.readline()
        data["bin_contents"] = []
        for line in fin.readlines():
            words = line.split()
            data["bin_contents"] += [float(word) for word in words]
        self.plot_data.append(data)

    def plots(self):
        self.one_d_plot()

    def one_d_plot(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        data_sum = []
        for item in self.plot_data:
            bins = item["energy"]["bins"]
            e_bin_width = item["energy"]["bin_width"]
            ang_bin_width = item["angular"]["bin_width"]
            data = [item["n_protons"]*ang_bin_width*e_bin_width*b for b in item["bin_contents"]]
            bin_centres = [(bins[i]+bins[i+1])/2.0 for i, b in enumerate(bins[1:])]
            axes.hist(bin_centres, weights=data, bins=bins, histtype="step", label="Item")
        data_sum = [item["n_protons"]*ang_bin_width*e_bin_width*b for b in self.data_sum]
        axes.hist(bin_centres, weights=data_sum, bins=bins, histtype="step", label="Sum")
        axes.legend()
        #axes.set_xlim(0, 0.005)
        axes.set_xlabel("Muon energy [GeV]")
        axes.set_ylabel("Number of muons")
        axes.set_title(f"Proton energy: {self.proton_energy:2.1f} GeV")

class PlotScan:
    def __init__(self):
        self.plot_fluka_yield_list = []

    def plot(self):
        min_e = 0.0035
        max_e = 0.0041
        mu_e_pos = []
        p_e_pos = []
        data = []
        data_sum = []
        data_peak = []
        for a_yield in self.plot_fluka_yield_list:
            norm = a_yield.plot_data[0]["n_protons"]*a_yield.plot_data[0]["energy"]["bin_width"]*a_yield.plot_data[0]["angular"]["bin_width"]
            data += [norm*b for b in a_yield.data_sum]
            mu_e_bins = a_yield.plot_data[0]["energy"]["bins"]
            min_bin = len([e for e in mu_e_bins if e < min_e])
            max_bin = len([e for e in mu_e_bins if e < max_e])
            mu_e_pos += [(mu_e_bins[i]+mu_e_bins[i+1])/2.0 for i, b in enumerate(mu_e_bins[1:])]
            p_e_pos += [a_yield.proton_energy for i, b in enumerate(mu_e_bins[1:])]
            data_sum.append(sum([norm*b for b in a_yield.data_sum[:max_bin]]))
            data_peak.append(sum([norm*b for b in a_yield.data_sum[min_bin:max_bin]]))
        mu_e_bins = self.plot_fluka_yield_list[0].plot_data[0]["energy"]["bins"]
        p_energy = [plot.proton_energy for plot in self.plot_fluka_yield_list]
        p_energy_bins = [(p_energy[i+1]+p_energy[i])/2 for i, p in enumerate(p_energy[1:])]
        p_energy_bins = [2*p_energy_bins[0]-p_energy_bins[1]]+p_energy_bins+[2*p_energy_bins[-1]-p_energy_bins[-2]]

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        h = axes.hist2d(mu_e_pos, p_e_pos, weights=data, bins=[mu_e_bins, p_energy_bins])
        axes.set_xlim(0.0, 0.005)
        axes.set_xlabel("Muon energy [GeV]")
        axes.set_ylabel("Proton energy [GeV]")
        figure.colorbar(h[3])
        figure.savefig(os.path.join(self.plot_dir, "yield_vs_energy_fluka_f23.png"))

        figure1d = matplotlib.pyplot.figure()
        axes1d = figure1d.add_subplot(1, 1, 1)
        h = axes1d.scatter(p_energy, data_sum, label = "Total with E < 4.1 MeV")
        h = axes1d.scatter(p_energy, data_peak, label = "Total with 3.5 < E < 4.1 MeV")
        axes1d.set_xlabel("Proton Energy [GeV]")
        axes1d.set_ylabel("Yield of $\\mu^+$")
        axes1d.legend()
        figure1d.savefig(os.path.join(self.plot_dir, "sum_yield_fluka_f23.png"))

def main():
    plot_fluka_yield_list = []
    for e in [0.1*i for i in range(4, 31, 1)]:
        try:
            fluka = PlotFlukaYield()
            fluka.proton_energy = e
            fluka.plot_dir = "output/fluka_model_v8/"
            fluka.file_glob = f"output/fluka_model_v8/energy_{e:2.1f}/plate_target*_fort.23"
            fluka.load_files()
            fluka.plots()
        except:
            continue
        plot_fluka_yield_list.append(fluka)
    scan = PlotScan()
    scan.plot_dir = plot_fluka_yield_list[0].plot_dir
    scan.plot_fluka_yield_list = plot_fluka_yield_list
    scan.plot()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
