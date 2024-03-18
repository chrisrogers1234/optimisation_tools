import shutil
import json
import glob
import os

import numpy
import matplotlib
import matplotlib.pyplot

import optimisation_tools.utils.utilities
import optimisation_tools.plotting.plot as plot

class PlotBumpScan():
    def __init__(self):
        self.b3 = "0.020"
        dir_root = "output/2023-03-01_baseline/bump_scan_v1/"
        self.dir_list = glob.glob(f"{dir_root}/b1*b3={self.b3}*/track_beam/reference/")
        self.plot_dir = f"{dir_root}/plots_b3={self.b3}"
        self.probe_files = ["ring_probe_001.h5", "ring_probe_009.h5", "ring_probe_013.h5"]
        self.track_data = {}
        self.field_data = {}

    def setup_plot_dir(self):
        if os.path.exists(self.plot_dir):
            shutil.rmtree(self.plot_dir)
        os.makedirs(self.plot_dir)

    def load_data(self):
        self.load_track_data()
        self.load_field_data()

    def load_track_data(self):
        self.track_data = dict([(f, []) for f in self.probe_files])
        for a_dir in self.dir_list:
            for a_file in self.probe_files:
                loader = plot.LoadH5(os.path.join(a_dir, a_file), verbose = 0)
                self.track_data[a_file].append(loader.data)

    def load_field_data(self):
        self.field_data = {}
        for a_dir in self.dir_list:
            a_file = os.path.join(a_dir, "subs.json")
            json_in = json.loads(open(a_file).read())
            for key in json_in.keys():
                if "bump" in key and "field" in key:
                    if key not in self.field_data:
                        self.field_data[key] = []
                    self.field_data[key].append(json_in[key])

    def plot(self, var_x, var_y, var_z_lambda, z_label, title=""):
        self.setup_plot_dir()
        n_dirs = len(self.dir_list)
        x_list = self.field_data[var_x]
        y_list = self.field_data[var_y]
        z_list = [var_z_lambda(i)  for i in range(n_dirs)]
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.set_xlabel(self.axis_labels[var_x])
        axes.set_ylabel(self.axis_labels[var_y])
        levels = numpy.linspace(min([0]+z_list), max(z_list), 10)
        contours = axes.tricontourf(x_list, y_list, z_list, levels=levels)
        if title:
            axes.set_title(title)
        axes.text(1.32, 1.0, z_label, transform=axes.transAxes, rotation="vertical", verticalalignment='top')
        min_index = z_list.index(min(z_list))
        figure.colorbar(contours)
        axes.scatter([x_list[min_index]], [y_list[min_index]], color="red")
        min_text = f"min: ({x_list[min_index]:3.3f}, {y_list[min_index]:3.3f}, {z_list[min_index]:4.4g})"
        axes.text(x_list[min_index], y_list[min_index], min_text, horizontalalignment='right', color='red')
        return figure

    def do_plots(self):
        get_miss_distance_1 = lambda i: (self.track_data["ring_probe_009.h5"][i][0]["r"] - self.track_data["ring_probe_001.h5"][i][0]["r"])
        fig = self.plot("__h_bump_4_field__", "__h_bump_5_field__", get_miss_distance_1, "$\\delta r_1$ [m]", f"B3 = {self.b3} T")
        fig.savefig(f"{self.plot_dir}/B4_vs_B5_vs_dr1.png")
        get_miss_distance_2 = lambda i: (self.track_data["ring_probe_013.h5"][i][0]["r"] - self.track_data["ring_probe_001.h5"][i][0]["r"])
        fig = self.plot("__h_bump_4_field__", "__h_bump_5_field__", get_miss_distance_2, "$\\delta r_2$ [m]", f"B3 = {self.b3} T")
        fig.savefig(f"{self.plot_dir}/B4_vs_B5_vs_dr2.png")
        get_miss_distance = lambda i: (get_miss_distance_1(i)**2+get_miss_distance_2(i)**2)**0.5
        fig = self.plot("__h_bump_4_field__", "__h_bump_5_field__", get_miss_distance, "$(\\delta r_1^2 + \\delta r_2^2)^{0.5}$ [m]", f"B3 = {self.b3} T")
        fig.savefig(f"{self.plot_dir}/B4_vs_B5_vs_dr_tot.png")

    axis_labels = {
        "__h_bump_3_field__":"B3 [T]",
        "__h_bump_4_field__":"B4 [T]",
        "__h_bump_5_field__":"B5 [T]"
    }

"""
Minimiser finds
     h bump 3 0.06
     h bump 4 0.20922542623698237
     h bump 5 0.2789160022609787
Orbit 14 |     3999.21804   0.0759298804              0              0     555.432104     3.00000024  |     737942.509  0.00134405977              0              0 |
Orbit 18 |     3995.86833   0.0751155967              0              0      831.92347      3.0000002  |     6203454.95 0.000813309621              0              0 |
"""

def main():
    plotter = PlotBumpScan()
    plotter.load_data()
    plotter.do_plots()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
