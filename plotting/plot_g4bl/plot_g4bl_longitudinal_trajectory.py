import datetime
import math
import json
import glob
import os
import sys

import numpy
numpy.set_printoptions(linewidth=200)
import matplotlib
import matplotlib.colors

from xboa.bunch import Bunch
from xboa.hit import Hit
import xboa.common

import optimisation_tools.opal_tracking
from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix

DecoupledTransferMatrix.det_tolerance = 1

class PlotG4BL(object):
    def __init__(self, run_dir_glob, co_file, reference_file, reference_file_format, plot_dir, max_score, label):
        self.plot_dir = plot_dir
        self.co_data = []
        self.frequency = 0.704
        self.station_to_z_dict = dict([(i, i*25-20000) for i in range(101)])
        self.load_data(run_dir_glob, co_file, reference_file, reference_file_format)
        self.parse_substitutions() # check for variables and sort the list
        self.filter_data(max_score) # if max_score, reject data with data["errors"] > max_score
        self.colors =  ["C"+str(i) for i in range(len(self.co_data))]
        self.event_list = [i for i in range(12)]
        self.e_lim = [-80.0, 300.0]
        self.t_lim = [-0.5/self.frequency, 0.5/self.frequency]
        self.emit_long = [3.61*self.mm2eVms, 3.61*self.mm2eVms*3]
        self.min_n_hits = 90
        self.label = label
        self.max_t_offset = 2.0
        self.will_plot_beam_ellipse = True
        utilities.clear_dir(plot_dir)

    def load_data(self, run_dir_glob, co_file, reference_file, reference_file_format):
        globble = []
        if type(run_dir_glob) == type([]):
            for run_dir in run_dir_glob:
                globble += glob.glob(run_dir)
        else:
            globble = glob.glob(run_dir_glob)
        if len(globble) == 0:
            raise RuntimeError(f"Failed to find any dirs from glob {run_dir_glob}")
        for a_dir in sorted(globble):
            print(globble)
            new_co_data = []
            reference_data_glob = sorted(glob.glob(os.path.join(a_dir, reference_file))) # output.txt
            for i, file_name in enumerate(reference_data_glob):
                new_co_data.append({"bunch_list":self.load_bunch(file_name, reference_file_format)})
            co_file_name = os.path.join(a_dir, co_file) # closed_orbits_cache
            try:
                fin = open(co_file_name)
                co_data = json.loads(open(co_file_name).read())
                if len(co_data) != len(new_co_data):
                    raise ValueError("Found", len(co_data), "transfer maps and", len(new_co_data), "reference outputs")
                for i, co in enumerate(co_data):
                    new_co_data[i].update(co)
            except OSError:
                print("Failed to load", co_file_name)
            self.co_data += new_co_data

    def load_bunch(self, file_name, reference_file_format):
        if reference_file_format == "g4bl_track_file":
            bunch_list = Bunch.new_list_from_read_builtin(reference_file_format, file_name)
        elif reference_file_format == "bdsim_root_file":
            analysis = optimisation_tools.opal_tracking.StoreDataInMemory()
            analysis.coordinate_transform = analysis.coord_dict["none"]
            analysis.dt_tolerance = -1
            analysis.station_dt_tolerance = -1
            optimisation_tools.opal_tracking.BDSIMTracking.read_files([file_name], analysis, self.station_to_z_dict, verbose=5)
            hit_list_of_lists = analysis.finalise()
            hits_by_station = analysis.sort_by_station(hit_list_of_lists)
            bunch_list = [Bunch.new_from_hits(hit_list) for hit_list in hits_by_station]
        else:
            raise RuntimeError(f"did not recognise file format {reference_file_format}")
        return bunch_list

    def _sort_lambda(self, data):
        if "variables" not in data:
            return []
        return [data["variables"][key] for key in self.key_list]

    def filter_data(self, max_score):
        if not max_score:
            return
        co_data_tmp = []
        for data in self.co_data:
            if "errors" not in data:
                continue
            if sum(data["errors"]) < max_score:
                co_data_tmp.append(data)
        self.co_data = co_data_tmp

    def get_label(self, data):
        label = ""
        for var, value in data["variables"].items():
            try:
                if self.key_subs[var] != None:
                    label += self.key_subs[var]+": "+str(value)+" "+self.units_subs[var]
            except KeyError:
                label += "KeyError"
        return label

    def parse_substitutions(self):
        subs_list = [data["substitutions"] for data in self.co_data if "substitutions" in data]
        self.key_list = list(utilities.dict_compare(subs_list))
        for data in self.co_data:
            if "substitutions" not in data:
                continue
            subs = data["substitutions"]
            var = dict([(key,subs[key]) for key in self.key_list if key in subs])
            data["variables"] = var

        self.co_data = sorted(self.co_data, key = self._sort_lambda)

    def do_plots(self):
        self.figure1 = matplotlib.pyplot.figure()
        self.figure2 = matplotlib.pyplot.figure()
        self.figure3 = matplotlib.pyplot.figure()
        self.plot_reference_z()
        self.figure1.savefig(self.plot_dir+"/z_vs_e.png")
        self.figure2.savefig(self.plot_dir+"/z_vs_t.png")
        self.figure3.savefig(self.plot_dir+"/t_vs_e.png")
        print("Plotted data with variables", self.key_list)
        max_keys = max([(len(data.keys()), data.keys()) for data in self.co_data])
        print("Top level data keys were", max_keys[1])

    def get_hit_list(self, event_id, bunch_list):
        hit_list = []
        for bunch in bunch_list:
            a_hit_list = bunch.get_hits("event_number", event_id)
            if len(a_hit_list):
                hit_list.append(a_hit_list[0])
            else:
                break
        return hit_list

    def plot_beam_ellipse(self, axes, x_list, y_list, n_theta):
        n_points = len(x_list)
        points = [[x_list[i], y_list[i]] for i in range(n_points)]
        mean, cov = xboa.common.fit_ellipse(points, 1e9)
        cov = cov/numpy.linalg.det(cov)**0.5
        covinv = numpy.linalg.inv(cov)
        print(f"Cov inv\n{covinv}")
        for emit in self.emit_long:
            plot_x_list, plot_y_list = [], []
            for i in range(n_theta+1):
                xvec_in = numpy.array([
                    math.cos(2*math.pi*i/n_theta),
                    math.sin(2*math.pi*i/n_theta)
                ])
                an_emit_in = numpy.dot(numpy.transpose(xvec_in), numpy.dot(covinv, xvec_in))
                xvec_out = xvec_in*(emit/an_emit_in)**0.5
                an_emit_out = numpy.dot(numpy.transpose(xvec_out), numpy.dot(covinv, xvec_out))
                #print(xvec_out, xvec_in, an_emit_in, an_emit_out)
                xvec_out += mean
                plot_x_list.append(xvec_out[0])
                plot_y_list.append(xvec_out[1])
            axes.plot(plot_x_list, plot_y_list, linestyle="dashed", color="r")

    def plot_reference_z(self):
        axes1 = self.figure1.add_subplot(1, 1, 1)
        axes2 = self.figure2.add_subplot(1, 1, 1)
        axes3 = self.figure3.add_subplot(1, 1, 1)
        for i, data in enumerate(self.co_data):
            bunch_list = data["bunch_list"]
            ref_list = self.get_hit_list(0, bunch_list)
            my_data = []
            for event_id in self.event_list:
                hit_list = self.get_hit_list(event_id, bunch_list)
                if len(hit_list) == 0:
                    print("No hits for event", event_id)
                    continue
                print("Event ID", event_id)
                for hit in hit_list:
                    print(f"{hit['station']} {hit['event_number']} {hit['z']} {hit['t']} {hit['energy']}")

                z_list = [hit["z"] for i, hit in enumerate(hit_list)]
                t_list = [hit["t"] for i, hit in enumerate(hit_list)] #-ref_list[i]["t"]
                e_list = [hit["energy"] for i, hit in enumerate(hit_list)] #-ref_list[i]["energy"]
                z_list = [z for i, z in enumerate(z_list)]# if abs(t_list[i]) < 2]
                e_list = [e for i, e in enumerate(e_list)]# if abs(t_list[i]) < 2]
                t_list = [t for t in t_list] # if abs(t) < self.max_t_offset]
                if len(t_list) > self.min_n_hits:
                    my_data.append({
                        "z_list":z_list,
                        "t_list":t_list,
                        "e_list":e_list,
                        "e_rms":numpy.std(e_list)
                    })
            my_data = sorted(my_data, key = lambda x: (-len(x["t_list"]), x["e_rms"]))
            if len(my_data) > 100:
                my_data = [my_data[0], my_data[int(len(my_data)/2)], my_data[-1]]
                my_data = [item for i, item in enumerate(my_data) if item not in my_data[i+1:]]
            for item in my_data:
                print("Found", len(hit_list), "hits for event", event_id, "with initial t, E", hit_list[0]["t"], hit_list[0]["energy"])
                c = self.colors[i]
                axes1.plot(item["z_list"], item["e_list"])#, c=c)
                axes2.plot(item["z_list"], item["t_list"])#, c=c)
                print(item["t_list"])
                if len(item["t_list"]) > self.min_n_hits:
                    axes3.scatter(item["t_list"], item["e_list"], c=[i for i in range(len(item["t_list"]))], s=2)
                else:
                    axes3.scatter(item["t_list"], item["e_list"], c="xkcd:light grey",s=2)
            if self.will_plot_beam_ellipse:
                if len(my_data) > 2:
                    self.plot_beam_ellipse(axes3, my_data[0]["t_list"], my_data[0]["e_list"], 8000)
                else:
                    self.plot_beam_ellipse(axes3, my_data[0]["t_list"], my_data[0]["e_list"], 8000)

        axes1.set_xlabel("z [mm]")
        axes1.set_ylabel("energy [MeV]")

        axes2.set_xlabel("z [mm]")
        axes2.set_ylabel("t [ns]")

        for x in [i*self.frequency/2.0 for i in range(-2, 3)]:
            axes3.plot([x, x], self.e_lim, c="grey", linestyle="dashed")
        axes3.set_xlabel("dt [ns]")
        axes3.set_ylabel("dE [MeV]")
        axes3.set_xlim(self.t_lim)
        #axes3.set_ylim(self.e_lim)

        for axes in axes1, axes2, axes3:
            axes.text(0.7, 0.85, self.label, transform=axes.transAxes)

    key_subs = {
            "__dipole_field__":"B$_{y}$",
            "__momentum__":"p$_{tot}$",
            "__wedge_opening_angle__":"$\\theta_{wedge}$",
            "__energy__":None,
    }
    units_subs = {
            "__dipole_field__":"[T]",
            "__momentum__":"[MeV/c]",
            "__wedge_opening_angle__":"$^\\circ$",
            "__energy__":"",
    }
    beta_limit = 1e4
    mm2eVms = 1000/2839.35 # mm 2 MeV ns (== eV ms)


def main():
    dt = "40"
    n_rf = "3"
    run_dir = "output/demo_v21/"
    run_dir_glob = [run_dir+f"dt={dt}*n_cavities={n_rf}*/"]
    run_dir_glob = [run_dir+f"pz=200*_version=2022*/"]
    plot_dir = run_dir+"/plot_longitudinal_trajectory_"+dt+"_"+n_rf+"/"

    file_name = "track_beam_amplitude/longitudinal/output*.root"
    co_file_name = "closed_orbits_cache"
    file_format = "icool_for009"
    file_format = "bdsim_root_file"
    label = f""+str(datetime.datetime.now())
    plotter = PlotG4BL(run_dir_glob, co_file_name, file_name, file_format, plot_dir, 1e9, label)
    plotter.beta_limit = 1e4
    plotter.event_list = [i for i in range(4)]
    plotter.min_n_hits = 1
    plotter.max_t_offset = 100
    plotter.will_plot_beam_ellipse = False
    plotter.do_plots()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")