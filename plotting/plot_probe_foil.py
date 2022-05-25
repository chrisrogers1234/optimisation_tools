import shutil
import glob
import os
import json
import sys
import operator
import math
import copy

import matplotlib
import numpy

import xboa.common
import xboa.hit

import config.config_base as config
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.twod_transfer_matrix import TwoDTransferMatrix
import optimisation_tools.utils.utilities
import optimisation_tools.plotting.plot

class PlotProbes(object):
    def __init__(self, pre_h5, foil_h5, post_h5, plot_dir, station_in, station_out):
        self.pre_h5 = pre_h5
        self.foil_h5 = foil_h5
        self.post_h5 = post_h5
        self.plot_dir = plot_dir
        self.station_in = station_in
        self.t_in = [-5, 5]
        self.station_out = station_out
        self.t_out = [110, 120]
        self.var_list = ["r", "r'", "z", "z'"]
        self.weight_list = [1.0, 0.01, 1.0, 0.01]
        self.score_leg = "$\\sqrt{x^2 + y^2 + \\frac{x'^2+y'^2}{0.01^2}}$ [mm]"
        self.bin_width = [0.01, 0.005] # x , y
        self.cmap = "PiYG"

    def get_ke(self, item):
        if item == None:
            return 0.0
        psquared = (item["px"]**2+item["py"]**2+item["pz"]**2)
        mass = xboa.common.pdg_pid_to_mass[2212]
        ke = (psquared + mass**2)**0.5 - mass
        return ke

    def get_plot_list(self):
        data_in = {}
        data_out = {}
        self.plot_list = []
        print("First ", self.pre_h5.data[0])
        for item in self.pre_h5.data:
            if item["station"] == self.station_in:
                data_in[item["id"]] = item
        for item in self.post_h5.data:
            if item["station"] == self.station_out:
                data_out[item["id"]] = item
        x_list, y_list, de_list, zp_list, rp_list = [], [], [], [], []

        for id0 in data_in:
            item_in = data_in[id0]
            print("Hit in t:", item_in["t"], "r:", item_in["r"], "phi:", item_in["phi"])
            if id0 in data_out:
                item_out = data_out[id0]
                print("Hit out t:", item_out["t"], "r:", item_out["r"], "phi:", item_out["phi"])
            else:
                item_out = None
            x_list.append(item_in["r"])
            y_list.append(item_in["z"])
            if item_out == None:
                de_list.append(0.1)
            else:
                de_list.append(abs(self.get_ke(item_out)-self.get_ke(item_in)))
                zp_list.append(item_out["z'"]-item_in["z'"])
                rp_list.append(item_out["r'"]-item_in["r'"])
            if id0 < 10:
                print(item_in["r"], item_in["z"], self.get_ke(item_in), self.get_ke(item_out), de_list[-1], item_in["z'"], item_out["z'"])
        return x_list, y_list, de_list, zp_list, rp_list

    def get_foil_list(self):
        nh_dict = {}
        r_list = []
        z_list = []
        for item in self.foil_h5.data:
            # we want to include events that only hit with initial injection
            if item["id"] not in nh_dict:
                nh_dict[item["id"]] = 0
            # we want to only count second strike so that the r-z plot is visible
            if item["station"] == 0:
                continue
            nh_dict[item["id"]] += 1
            r_list.append(item["r"])
            z_list.append(item["z"])
        nh_list = [v for v in nh_dict.values()]
        return nh_list, r_list, z_list

    def get_bins(self, x_list, bin_width):
        n_bins = int((max(x_list)-min(x_list))/bin_width)+2
        bin_list = [min(x_list)+(i-0.5)*bin_width for i in range(n_bins)]
        return bin_list

    def plot_change(self):
        x_list, y_list, de_list, zp_list, rp_list = self.get_plot_list()
        x_bin_list = self.get_bins(x_list, self.bin_width[0])
        y_bin_list = self.get_bins(y_list, self.bin_width[1])
        print(x_bin_list)
        print(y_bin_list)
        print(x_list)
        print(y_list)
        print(de_list)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        min_z = min(de_list)
        max_z = max(de_list)
        hout = axes.scatter(x_list, y_list,  c=de_list)
        axes.set_xlabel("r [m]")
        axes.set_ylabel("z [m]")
        axes.text(1.0, 1.05, "dE [MeV]", transform=axes.transAxes)
        figure.colorbar(hout)
        figure.savefig(os.path.join(self.plot_dir, "dedx.png"))

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 2, 1)
        hout = axes.hist(rp_list, bins=100)
        axes.text(0.05, 0.95, "RMS r': "+format(numpy.std(rp_list), "4.3g"), transform=axes.transAxes)
        axes.set_xlabel("r'")
        axes = figure.add_subplot(1, 2, 2)
        hout = axes.hist(zp_list, bins=100)
        axes.text(0.05, 0.95, "RMS z': "+format(numpy.std(zp_list), "4.3g"), transform=axes.transAxes)
        axes.set_xlabel("z'")
        figure.savefig(os.path.join(self.plot_dir, "scattering.png"))


    def plot_foil(self):
        nh_list, r_list, z_list = self.get_foil_list()
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        hout = axes.hist(nh_list, bins=max(nh_list)+1)
        axes.set_xlim(-0.5, max(nh_list)+0.5)
        axes.text(0.75, 0.95, "Mean N: "+format(numpy.mean(nh_list), "4.3g"), transform=axes.transAxes)
        axes.set_xlabel("Number of hits")
        figure.savefig(os.path.join(self.plot_dir, "n_hits.png"))

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        hout = axes.hist2d(r_list, z_list, bins=[100, 100])
        axes.set_xlabel("r [m]")
        axes.set_xlabel("z [m]")
        figure.savefig(os.path.join(self.plot_dir, "hit_r-z.png"))


def clear_plot_dir(plot_dir):
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

def main():
    DecoupledTransferMatrix.det_tolerance = 1.0
    for foil_dir in ["injected_beam/", "scattering_test/", "de_test/"]:
        input_dir = os.path.join("output/2022-03-01_baseline/correlated_painting/tracking_v15/track_beam/", foil_dir)
        plot_dir = input_dir+"/plot_foil/"
        clear_plot_dir(plot_dir)
        plotter = PlotProbes(
            optimisation_tools.plotting.plot.LoadH5(input_dir+"PREFOILPROBE.h5"),
            optimisation_tools.plotting.plot.LoadH5(input_dir+"FOIL.h5"),
            optimisation_tools.plotting.plot.LoadH5(input_dir+"POSTFOILPROBE.h5"),
            plot_dir, 0, 0)
        plotter.plot_change()
        plotter.plot_foil()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Done")
