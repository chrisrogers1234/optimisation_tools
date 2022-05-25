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
import opal_tracking

import xboa.common
import xboa.hit

import config.config_double_triplet_baseline as config
from utils.decoupled_transfer_matrix import DecoupledTransferMatrix
import utils.utilities
import plot

class PlotProbes(object):
    def __init__(self, load_h5, plot_dir, station_in, station_out):
        self.load_h5 = load_h5
        self.plot_dir = plot_dir
        self.station_in = station_in
        self.t_in = [-5, 5]
        self.station_out = station_out
        self.t_out = [110, 120]
        self.var_list = ["r", "r'", "z", "z'"]
        self.weight_list = [1.0, 0.01, 1.0, 0.01]
        self.score_leg = "$\\sqrt{x^2 + y^2 + \\frac{x'^2+y'^2}{0.01^2}}$ [mm]"
        self.bin_width = [0.050, 0.050] # x , y
        self.cmap = "PiYG"
        self.max_z = None


    def score_function(self, item_in, item_out):
        if item_out == None:
            return -1.0
        score = [((item_out[var] - item_in[var])**2/self.weight_list[i]**2)**0.5 \
                                                       for i, var in enumerate(self.var_list)]       
        print("Item in:", [item_in[var] for var in self.var_list], "out:", [item_out[var] for var in self.var_list])
        score = sum(score)
        if self.max_z != None and score > self.max_z:
            score = self.max_z
        return score

    def swap_negative(self, z, z0):
        if z < 0.0:
            return 0.0
        return z

    def get_plot_list(self):
        data_in = {}
        data_out = {}
        self.plot_list = []
        print("First ", self.load_h5.data[0])
        for item in self.load_h5.data:
            if item["station"] == self.station_in:
                if item["t"] < self.t_in[0] or item["t"] > self.t_in[1]:
                    continue
                data_in[item["id"]] = item
            elif item["station"] == self.station_out:
                data_out[item["id"]] = item
            else:
                continue
        x_list, y_list, z_list = [], [], []

        for id0 in data_in:
            item_in = data_in[id0]
            if id0 in data_out:
                item_out = data_out[id0]
            else:
                item_out = None
            x_list.append(item_in["x"])
            y_list.append(item_in["z"])
            z_list.append(self.score_function(item_in, item_out))
        z0 = -max(z_list)
        z_list = [self.swap_negative(z, z0) for z in z_list]
        return x_list, y_list, z_list

    def get_bins(self, x_list, bin_width):
        n_bins = int((max(x_list)-min(x_list))/bin_width)+2
        bin_list = [min(x_list)+(i-0.5)*bin_width for i in range(n_bins)]
        return bin_list

    def plot(self):
        x_list, y_list, z_list = self.get_plot_list()
        x_bin_list = self.get_bins(x_list, self.bin_width[0])
        y_bin_list = self.get_bins(y_list, self.bin_width[1])
        print(x_bin_list)
        print(y_bin_list)
        print(x_list)
        print(y_list)
        print(z_list)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        min_z = min(z_list)
        max_z = max(z_list)
        vtot = max(abs(min_z), abs(max_z))
        hout = axes.hist2d(x_list, y_list, bins=[x_bin_list, y_bin_list], weights=z_list,
                           cmin=min_z, cmax=max_z, cmap=self.cmap, vmin=1e-1, vmax=vtot, norm = matplotlib.colors.LogNorm())
        axes.set_title(self.score_leg)
        figure.colorbar(hout[3], ax=axes)
        figure.savefig(os.path.join(self.plot_dir, "out_distance.png"))


def clear_plot_dir(plot_dir):
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

def main():
    DecoupledTransferMatrix.det_tolerance = 1.0
    input_dir = "output/arctan_baseline/isis2_baseline/track_beam/grid/"
    plot_dir = input_dir+"/plot_probe_grid/"
    clear_plot_dir(plot_dir)
    plotter = PlotProbes(plot.LoadH5(input_dir+"RINGPROBE0?.h5"), plot_dir, 0, 1)
    plotter.max_z = 10
    plotter.plot()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Done")
