import shutil
import json
import os
import glob
import operator


import numpy
import matplotlib

import xboa.bunch
from xboa.bunch import Bunch

from optimisation_tools.utils import utilities


class PlotG4BLEmittance(object):
    def __init__(self, run_dir_glob, file_name_glob, file_format, plot_dir):
        self.plot_dir_root = plot_dir
        self.short_keys_of_interest = []
        self.data = []
        self.px_range = [-100.0, 100.0]
        self.x_range = [-100.0, 100.0]
        self.pz_range = [150.0, 250.0]
        self.a_range = [-1.0, 50.0]
        self.e_range = [150.0, 250.0]
        self.emit_range = [0.0, 5.0]
        self.emit6d_range = [0.0, 0.1]
        self.ct_range = [-20*300.0, 100*300.0]
        self.z_range = [None, None]
        self.file_format = file_format
        self.colors = []
        self.title = ""
        self.max_station = 10000
        self.frequency = 0
        self.output_filename = "performance.json"
        self.analysis_list = None
        self.run_dir_glob = run_dir_glob
        self.file_name_glob = file_name_glob
        self.e_min = 0
        self.e_max = 1e3
        self.amp_trans_max = 30
        self.amp_long_max = 30
        self.psize = 3


    def glob_data(self):
        print("Globbing", self.run_dir_glob)
        glob_list = sorted(glob.glob(self.run_dir_glob))
        for a_dir in glob_list:
            print("... searching", a_dir)
            for a_file in glob.glob(os.path.join(a_dir, self.file_name_glob)):
                my_data = {"file_name":a_file, "bunch_list":[], "plot_dir":os.path.join(a_dir, self.plot_dir_root)}
                my_data["substitutions"] = json.load(open(os.path.join(a_dir, "subs_list.json")))
                my_data["bunch_list"] = Bunch.new_list_from_read_builtin(self.file_format, my_data["file_name"])
                self.data.append(my_data)
        if len(glob_list) == 0:
            ls_target = os.path.split(self.run_dir_glob[:-1])[0]
            dir_ls = os.listdir(ls_target)
            for fname in sorted(dir_ls):
                print(fname)
        self.colors =  ["C"+str(i) for i in range(len(self.data))]

    def do_plots(self):
        for item in self.data:
            utilities.clear_dir(item["plot_dir"])
            self.plot_dt(item, 0, 10)
            self.plot_dt(item, 0, 20)

    def singles_cuts(self, bunch, test_bunch):
        """
        Return a bunch with hits that are a subset of test_bunch,
        with unique event_number and sorted by event_number
        """
        bunch = Bunch.new_from_hits(sorted(bunch.hits(), key=lambda h: h["event_number"]))
        ev_list = bunch.list_get_hit_variable(["event_number"])[0]
        sorted_ev, unique_elements = numpy.unique(ev_list, True)
        for hit in bunch:
            hit["local_weight"] = 0.0
        for el in unique_elements:
            bunch[el]["local_weight"] = 1.0
        bunch.transmission_cut(test_bunch, True)
        bunch.conditional_remove({"weight":0.5}, operator.lt)
        return bunch

    def plot_dt(self, item, s0, s1):
        plot_dir = item["plot_dir"]
        slice_p0 = 195
        slice_p1 = 205
        centile = 0.95 # color_axis range
        bunch0 = item["bunch_list"][s0]
        bunch1 = item["bunch_list"][s1]
        print("N events before cuts", len(bunch0), len(bunch1))

        bunch0 = self.singles_cuts(bunch0, bunch1)
        bunch1 = self.singles_cuts(bunch1, bunch0)

        z0 = bunch0[0]["z"]
        z1 = bunch1[0]["z"]
        print("N events after cuts", len(bunch0), len(bunch1))
        print("plot dt from z", z0, "to", z1)

        data_list = ["t", "amplitude x y", "p"]
        data0 = bunch0.list_get_hit_variable(data_list)
        data1 = bunch1.list_get_hit_variable(data_list)

        delta_t_list = [(data1[0][i]-data0[0][i]) for i, t0 in enumerate(data0[0])]
        index_centile = int(len(delta_t_list)*centile)
        value_centile = sorted(delta_t_list)[index_centile]

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        scatter = axes.scatter(data0[2], data0[1], c=delta_t_list, s=2, vmax=value_centile)
        figure.colorbar(scatter)
        axes.text(0.0, 1.025, f"Tracking from {int(z0)} to {int(z1)} mm", transform=axes.transAxes)
        axes.text(1.22, 0.68, f"Time-of-flight [ns]", transform=axes.transAxes, rotation=90)
        axes.set_xlabel("Momentum [MeV/c]")
        axes.set_ylabel("Amplitude [mm]")
        figure.savefig(f"{plot_dir}/amplitude_p_dt_{int(z0)}_{int(z1)}.png")


        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.hist(data0[1], 100)
        axes.text(0.7, 0.95, f"Mean {numpy.mean(data0[1]):8.4g} mm", transform=axes.transAxes)
        axes.text(0.0, 1.025, f"Tracking from {int(z0)} to {int(z1)} mm", transform=axes.transAxes)
        axes.set_xlabel("Amplitude [mm]")
        figure.savefig(f"{plot_dir}/amplitude_{int(z0)}_{int(z1)}.png")

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        p_cut = lambda p: p < slice_p1 and p > slice_p0
        amp_cut = [amp for i, amp in enumerate(data0[1]) if p_cut(data0[2][i])]
        dt_cut = [dt for i, dt in enumerate(delta_t_list) if p_cut(data0[2][i])]
        axes.text(0.0, 1.025, f"Tracking from {int(z0)} to {int(z1)} mm; considering {slice_p0} < p < {slice_p1} MeV/c", transform=axes.transAxes)
        axes.scatter(dt_cut, amp_cut, s=2)
        axes.set_xlim([min(dt_cut), value_centile])
        axes.set_xlabel("Time-of-flight [ns]")
        axes.set_ylabel("Amplitude [mm]")
        figure.savefig(f"{plot_dir}/slice_amp_vs_dt_{int(z0)}_{int(z1)}.png")


def make_run_dir_glob(run_dir, variables):
    a_glob = run_dir+"/name=cooling;_"
    for key, value in variables:
        a_glob += key+"="+value+";_"
    a_glob = a_glob[:-2]+"/"
    return a_glob

def main():
    tracking = "beam"
    variables = [
        ("by", "*"),
        ("dt", "*"),
        ("dtheta", "*"),
        ("efield", "*"),
        ("gap", "*"),
        ("iris_factor", "*"),
        ("pz", "*"),
        ("roll", "270"),
        ("window_dt", "*"),
    ]
    run_dir = "output/v060/"
    run_dir_glob = make_run_dir_glob(run_dir, variables)
    optimisation_dir = "test_a-pz_t"
    plot_dir = f"{optimisation_dir}_plots/"
    file_name = f"tmp/optimisation/{optimisation_dir}/output.txt"
    file_format = "icool_for009"
    frequency = 0.704
    print(f"Running with glob '{run_dir_glob}'")
    plotter = PlotG4BLEmittance(run_dir_glob, file_name, file_format, plot_dir)
    plotter.glob_data()
    plotter.do_plots()
    return

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
