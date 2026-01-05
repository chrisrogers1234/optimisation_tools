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

class Cut(object):
    def __init__(self, variable, value, function):
        self.variable = variable
        self.value = value
        self.function = function

    def will_cut(self, hit, decoupled, aa, ref_hit):
        if self.variable == "au" or self.variable == "av":
            index = {"au":1, "av":3}[self.variable]
            return self.function(aa[index], self.value)
        return self.function(hit[self.variable], self.value)

class Cut2(object):
    def __init__(self, variable, value, function, test_hit_list):
        self.rejected = []
        for hit in test_hit_list:
            test_value = hit[variable]
            if function(test_value, value):
                print(hit[-1], test_value, value)
                self.rejected.append(hit[-1])
        #set([hit[-1] for hit in test_hit_list \
        #                                      if function(hit[variable], value)])
        self.rejected = set(self.rejected)
        print("Cutting on", variable, "value", value, "gives:", self.rejected)

    def will_cut(self, hit, decoupled, aa, ref_hit):
        return hit['event_number'] in self.rejected

class RefCut(object):
    def __init__(self, variable, value, function):
        self.variable = variable
        self.value = value
        self.function = function

    def will_cut(self, hit, decoupled, aa, ref_hit):
        return self.function(hit[self.variable]-ref_hit[self.variable], self.value)

class TransmissionCut(object):
    def __init__(self, accept_hit_list):
        self.accepted = set([hit[-1] for hit in accept_hit_list])

    def will_cut(self, hit, decoupled, aa, ref_hit):
        return hit['event_number'] not in self.accepted

class PlotProbes(object):
    def __init__(self, setup_list, plot_dir):
        path = os.path.expandvars("${OPAL_EXE_PATH}/opal")
        mass = xboa.common.pdg_pid_to_mass[2212]
        ref = xboa.hit.Hit.new_from_dict({"pid":2212, "mass":mass, "charge":1.})
        my_config = config.Config()
        my_config.tracking["verbose"] = 100
        my_config.tracking["station_dt_tolerance"] = 1000.0
        my_config.tracking["analysis_coordinate_system"] = "none"
        self.data = copy.deepcopy(setup_list)

        for item in self.data:
            item["probe_data"] = opal_tracking._opal_tracking.StoreDataInMemory(my_config)
            tracking = opal_tracking.OpalTracking("", "", ref, item["file_name_list"], path)
            tracking.set_file_format("hdf5")
            tracking.pass_through_analysis = item["probe_data"]
            item["tracking"] = tracking
            item["stations"] = {}

        self.plot_dir = plot_dir
        self.fig_list = []
        self.shared_range = []
        self.s = 5

        self.id_cut = [0]
        self.time_window = 100.0
        self.height_window = [-200.0, 200.0]

    def load_data(self):
        for item in self.data:
            item["tracking"]._read_probes()
            stations = {}
            for hit_list in item["probe_data"].last:
                for hit in hit_list:
                    if hit["station"] not in stations:
                        stations[hit["station"]] = 0
                    stations[hit["station"]] += 1
            item["stations"] = stations
            print("  ... loaded following stations:number of hits", json.dumps(stations, indent=2))

    def get_hits(self, item):
        hit_list_of_lists = item["probe_data"].last
        item["hit_list"] = []
        for hit_list in hit_list_of_lists:
            for hit in hit_list:
                if hit["station"] == item["station"]:
                    item["hit_list"].append(hit)
                    for p in ["px", "py", "pz"]:
                        hit[p] = hit[p]*item["momentum_scale"]
        item["hit_list"] = self.cuts(item["hit_list"])
        return item["hit_list"]

    def cuts(self, data):
        if self.id_cut:
            data = [item for item in data if item["event_number"] not in self.id_cut]
        if self.time_window:
            start_time = min([item["t"] for item in data])
            data = [item for item in data if item["t"] < start_time+self.time_window]
        if self.height_window:
            data = [item for item in data if item["y"] < self.height_window[1] and item["y"] > self.height_window[0]]
        return data


    def plot(self):
        station_list = sorted(list(self.data[0]["stations"].keys()))
        station_list = station_list[0:2:1]#+station_list[0:11:1]+station_list[60::10]+station_list[-2:-1:1]
        station_list = set(station_list)
        for station in station_list:
            for item in self.data:
                self.get_hits(item)
            if sum([len(item["hit_list"]) for item in self.data]) == 0:
                print("No data for station", station)
                continue
            z_axis = None
            xlim, ylim = None, None
            figure = matplotlib.pyplot.figure(figsize=(20,10))
            axes = figure.add_subplot(2, 2, 1)
            self.plot_phase_space(axes, "x", "px", z_axis, station)
            axes = figure.add_subplot(2, 2, 2)
            self.plot_phase_space(axes, "y", "py", z_axis, station)
            axes = figure.add_subplot(2, 2, 3)
            self.plot_phase_space(axes, "x", "y", z_axis, station)
            axes = figure.add_subplot(2, 2, 4)
            self.plot_phase_space(axes, "px", "py", z_axis, station)
            axes.legend()
            figure.suptitle(self.name)
            fname = self.plot_dir+"/physical-space_station-"+self.name.replace(" ", "_")+"_"+str(station)+".png"
            figure.savefig(fname)
            print("Saved to", fname)
            self.fig_list.append(figure)


    def plot_phase_space(self, axes, x_axis, y_axis, z_axis, station):
        labels = {
            "kinetic_energy":"KE [MeV]",
            "x":"x [mm]",
            "y":"y [mm]",
            "px":"p$_x$ [MeV/c]",
            "py":"p$_y$ [MeV/c]",
            "x'":"x'",
            "y'":"y'",
            "t":"time [ns]",
            0:"station",
            1:"x [mm]",
            2:"p$_x$ [MeV/c]",
            3:"y [mm]",
            4:"p$_y$ [MeV/c]",
            5:"u",
            6:"p$_u$",
            7:"v",
            8:"p$_v$",
            9:"$\\phi_u$",
            10:"Norm. A$_u$ [mm]",
            11:"$\\phi_v$",
            12:"Norm. A$_v$ [mm]",
        }
        if True:
            labels[2] = "x'"
            labels[4] = "y'"
            labels[6] = "u'"
            labels[8] = "v'"
        name = "phase_space_"+str(station).rjust(3, "0")+"_"+str(x_axis)+"_"+str(y_axis)
        for item in self.data:
            if len(item["hit_list"]) == 0:
                continue
            x_list = [hit[x_axis] for hit in item["hit_list"]]
            y_list = [hit[y_axis] for hit in item["hit_list"]]
            scat = axes.scatter(x_list, y_list, c=item["color"], s=self.s, label=item["name"])
        print("Plotted", [len(item["hit_list"]) for item in self.data], "points")
        axes.set_xlabel(labels[x_axis])
        axes.set_ylabel(labels[y_axis])
        if z_axis != None:
            axes.get_figure().colorbar(scat)

def main_ramp_fields():
    DecoupledTransferMatrix.det_tolerance = 1.0
    dir_1 = "output/arctan_baseline/ramp_fields_low_amplitude//track_beam/forwards/"
    plot_dir = dir_1+"/plot_probe_physical/"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    probename = "RINGPROBE01.h5"
    probe_list = [{
        "file_name_list":glob.glob(dir_1+probename),
        "color":"blue",
        "momentum_scale":1,
        "station":station,
        "name":"ringprobe01"
    } for station in range(41)]
    plotter = PlotProbes(probe_list, plot_dir)
    plotter.name = "name"
    plotter.co_param_list = [{
        "filename":os.path.join(dir_1, "../../closed_orbits_cache"),
        "ref_to_bump_station_mapping":dict([(i,i) for i in range(1001)]),
    },]
    try:
        plotter.load_data()
    except IOError:
        print("IOError trying", probename)
        raise
    plotter.plot()


def main():
    DecoupledTransferMatrix.det_tolerance = 1.0
    dir_1 = "output/arctan_baseline/bump_quest/track_v2_bump_r_40_theta_180/track_beam/"
    dir_2 = "output/arctan_baseline/bump_quest/track_v2_bump_r_-1_theta_0/track_beam/"
    output_dir = dir_1.split("/tmp/")[0]
    plot_dir = output_dir+"/plot_probe_physical/"
    config = dir_1.split("/tmp/")
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    for probe, name in [("1", "foil"), ("2", "septum")]: #range(1, 3):
        probename = "FOILPROBE_"+probe+".h5"
        forwards = {
            "file_name_list":glob.glob(dir_1+"forwards/"+probename),
            "color":"blue",
            "momentum_scale":1,
            "station":0,
            "name":"bumped H$^{+}$"
        }
        backwards = {
            "file_name_list":glob.glob(dir_1+"backwards/"+probename),
            "color":"orange",
            "momentum_scale":-1,
            "station":0,
            "name":"injected H$^{-}$"
        }
        reference = {
            "file_name_list":glob.glob(dir_2+"forwards/"+probename),
            "color":"green",
            "momentum_scale":1,
            "station":0,
            "name":"unbumped"
        }
        plotter = PlotProbes([forwards, reference], plot_dir)
        plotter.name = name
        plotter.co_param_list = [{
            "filename":os.path.join(dir_1, "../closed_orbits_cache"),
            "ref_to_bump_station_mapping":dict([(i,i) for i in range(1001)]),
        },]
        try:
            plotter.load_data()
        except IOError:
            print("IOError trying", probename)
            raise
        plotter.plot()

if __name__ == "__main__":
    main_ramp_fields()
    matplotlib.pyplot.show(block=False)
    input("Done")
