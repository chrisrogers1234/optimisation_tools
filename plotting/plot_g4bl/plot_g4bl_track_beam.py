import math
import json
import glob
import os
import sys

import scipy.stats
import numpy
numpy.set_printoptions(linewidth=200)
import matplotlib
import matplotlib.colors

from xboa.bunch import Bunch
from xboa.hit import Hit
import xboa.common

from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix

DecoupledTransferMatrix.det_tolerance = 1

class PlotG4BL(object):
    def __init__(self, run_dir_glob, co_file, cell_length, reference_file, reference_file_format, plot_dir, max_score):
        self.plot_dir = plot_dir
        self.co_data = []
        self.cell_length = cell_length
        self.variables_of_interest = None
        self.colors =  ["C"+str(i) for i in range(len(self.co_data))]
        self.a_4d_max = 1e9
        self.max_score = max_score
        self.title = ""
        self.n_a4d_bins = 40
        self.load_data(run_dir_glob, co_file, reference_file, reference_file_format)

    def load_data(self, run_dir_glob, co_file, reference_file, reference_file_format):
        for a_dir in sorted(glob.glob(run_dir_glob)):
            new_co_data = []
            reference_data_glob = sorted(glob.glob(os.path.join(a_dir, reference_file)))
            if len(reference_data_glob) == 0:
                print("Failed to find files from dir ", a_dir)
            for i, file_name in enumerate(reference_data_glob):
                new_co_data.append({"bunch_list":Bunch.new_list_from_read_builtin(reference_file_format, file_name)})
                print("   ... found ", len(new_co_data[-1]["bunch_list"][-1]), "in final bunch")
            co_file_name = os.path.join(a_dir, co_file)
            try:
                fin = open(co_file_name)
                co_data = json.loads(open(co_file_name).read())
                if len(co_data) != len(new_co_data):
                    raise ValueError("Found", len(co_data), "transfer maps and", len(new_co_data), "reference outputs")
                for i, co in enumerate(co_data):
                    new_co_data[i].update(co)
            except (OSError, ValueError):
                sys.excepthook(*sys.exc_info())
                print("Failed to load", co_file_name)
            self.co_data += new_co_data
            print("Done", a_dir)
        if not len(self.co_data):
            print("Failed to find any files using: {0}".format(run_dir_glob))

    def _sort_lambda(self, data):
        if "variables" not in data:
            return []
        return [data["variables"][key] for key in self.key_list]

    def parse_substitutions(self):
        subs_list = [data["substitutions"] for data in self.co_data if "substitutions" in data]
        self.key_list = list(utilities.dict_compare(subs_list))
        my_lambda = lambda key: self.variables_of_interest == None or key in self.variables_of_interest
        self.key_list = [key for key in self.key_list if my_lambda(key)]
        for data in self.co_data:
            if "substitutions" not in data:
                continue
            subs = data["substitutions"]
            var = dict([(key,subs[key]) for key in self.key_list if key in subs])
            data["variables"] = var

        self.co_data = sorted(self.co_data, key = self._sort_lambda)

    def filter_data(self):
        if not self.max_score:
            return
        co_data_tmp = []
        for data in self.co_data:
            if "errors" not in data:
                continue
            if sum(data["errors"]) < self.max_score:
                co_data_tmp.append(data)
            else:
                print("Filter", data["variables"], "with score", sum(data["errors"]))
        self.co_data = co_data_tmp

    def get_tm(self, tracking_matrix):
        if tracking_matrix == None:
            return numpy.zeros((4,4))
        tm = numpy.array([row[1:5] for row in tracking_matrix[0:4]])
        tm = DecoupledTransferMatrix.simplectify(tm)
        return tm

    def get_amplitude(self, hit, ref_hit, tm):
        var_list = ["x", "x'", "y", "y'"]
        try:
            seed = [hit[var]-ref_hit[var] for var in var_list]
        except ZeroDivisionError:
            return
        try:
            aa = tm.coupled_to_action_angle(seed)
        except (TypeError, numpy.linalg.LinAlgError):
            sys.excepthook(*sys.exc_info())
            return
        aa[1] *= ref_hit["p"]/ref_hit["mass"]
        aa[3] *= ref_hit["p"]/ref_hit["mass"]

        return aa

    def do_plots(self, input_station, output_station):
        self.parse_substitutions() # check for variables and sort the list
        self.filter_data() # if max_score, reject data with data["errors"] > max_score
        utilities.clear_dir(self.plot_dir)
        variables, a_4d_good, a_4d_all = self.plot_phase_space(input_station, output_station)
        self.plot_a4d_ratio(variables, a_4d_good, a_4d_all)

    def plot_phase_space(self, input_station, output_station):
        input_bunch, output_bunch = xboa.bunch.Bunch(), xboa.bunch.Bunch()
        a_4d_all = []
        a_4d_good = []
        variables = []
        tm = None #numpy.identity(4)
        for data in self.co_data:
            variables.append(data["variables"])
            for bunch in data["bunch_list"]:
                if bunch[0]["station"] == input_station:
                    input_bunch = bunch
                if bunch[0]["station"] == output_station:
                    output_bunch = bunch
            print("    with", variables[-1], "found", len(input_bunch), "input events", len(output_bunch), "output events")
            good_events = [hit["event_number"] for hit in output_bunch]

            figure = matplotlib.pyplot.figure(figsize=(20,10))
            axes = figure.add_subplot(2, 2, 1)

            x_list = [hit["x"] for hit in input_bunch]
            px_list = [hit["px"] for hit in input_bunch]
            axes.scatter(x_list, px_list, c="grey")

            x_list = [hit["x"] for hit in input_bunch if hit["event_number"] in good_events]
            px_list = [hit["px"] for hit in input_bunch if hit["event_number"] in good_events]
            z_list = [hit["z"] for hit in output_bunch]
            axes.scatter(x_list, px_list, c="blue")

            axes.set_xlabel("x [mm]")
            axes.set_ylabel("p$_x$ [MeV/c]")

            axes = figure.add_subplot(2, 2, 2)
            y_list = [hit["y"] for hit in input_bunch]
            py_list = [hit["py"] for hit in input_bunch]
            axes.scatter(y_list, py_list, c="grey")

            y_list = [hit["y"] for hit in input_bunch if hit["event_number"] in good_events]
            py_list = [hit["py"] for hit in input_bunch if hit["event_number"] in good_events]
            axes.scatter(y_list, py_list, c="blue")

            axes.set_xlabel("y [mm]")
            axes.set_ylabel("p$_y$ [MeV/c]")
            try:
                print(self.get_tm(data["tm"]))
                tm = DecoupledTransferMatrix(self.get_tm(data["tm"]))
            except ValueError: # just try to plug on
                tm = None
            if tm is None or tm.chol is None:
                print("Error for", data["variables"], "using default tm")
                tm = DecoupledTransferMatrix(self.default_tm)

            ref_hit = xboa.hit.Hit.new_from_dict(data["ref_track"][0])

            axes = figure.add_subplot(2, 2, 3)
            amplitude_list = [self.get_amplitude(hit, ref_hit, tm) for hit in input_bunch]
            print("Amplitude list length", len(amplitude_list))
            amplitude_list = [a for a in amplitude_list if a != None]
            print("Amplitude list length cutting None", len(amplitude_list))
            au_list = [aa[1] for aa in amplitude_list]
            av_list = [aa[3] for aa in amplitude_list]
            a_4d_all.append([aa[1]+aa[3] for aa in amplitude_list])
            axes.scatter(au_list, av_list, c="grey")
            amplitude_list = [self.get_amplitude(hit, ref_hit, tm) for hit in input_bunch if hit["event_number"] in good_events]
            amplitude_list = [a for a in amplitude_list if a != None]
            au_list = [aa[1] for aa in amplitude_list]
            av_list = [aa[3] for aa in amplitude_list]
            a_4d_good.append([aa[1]+aa[3] for aa in amplitude_list])
            title = self.get_title(data)
            axes.set_title(title)
            axes.scatter(au_list, av_list, c="blue")
            axes.set_xlabel("A$_{u}$ [mm]")
            axes.set_ylabel("A$_{v}$ [mm]")

            name = [self.short_form[key]+"_"+str(data["substitutions"][key]) for key in self.variables_of_interest]
            figure.savefig(os.path.join(self.plot_dir, "input_station_"+"_".join(name)+".png"))
            if len(matplotlib.pyplot.get_fignums()) > 0:
                matplotlib.pyplot.close(figure)
        return variables, a_4d_good, a_4d_all

    def get_title(self, data=None):
        if data == None:
            key_list = []
            for key in self.variables_of_interest:
                values = [data["substitutions"][key] for data in self.co_data]
                if numpy.std(values) < 1e-12:
                    key_list.append(key)
                print(key, key_list, numpy.std(values))
            data = self.co_data[0]
        else:
            key_list = self.variables_of_interest

        title_str = ""
        if self.title != "":
            title_str = self.title+"\n"+title_str
        for key in key_list:
            sub = round(data["substitutions"][key], 3)
            title_str += self.key_subs[key]+" "+str(sub)+" "+self.units_subs[key]+"; "
        return title_str

    def plot_beta(self, axes):
        x_key = self.key_list[0]
        x_list = []
        y_list = []
        for data in self.co_data:
            x_list.append(data["variables"][x_key])
            try:
                tm = DecoupledTransferMatrix(self.get_tm(data["tm"]))
                beta = tm.get_beta(0)
            except Exception:
                beta = 0.0
            if beta < self.beta_limit:
                y_list.append(beta)
            else:
                y_list.append(0.0)
        axes.plot(x_list, y_list, "o-", label="Estimated $\\beta_{\\perp}$ [mm]", c="grey")
        axes.set_ylabel("$\\beta_{\\perp}$ [mm]", fontsize=20)
        axes.tick_params(labelsize=14)
        axes.set_ylim(0, 4000.0)

    def plot_eps_eqm(self, axes, centile):
        x_key = self.key_list[0]
        x_list = []
        y_list = []
        beta = 0.0
        mass = xboa.common.pdg_pid_to_mass[13]
        dedx = 0.1555 # LiH minimum ionising, MeV/mm
        l_r = 970.9 # LiH radiation length, mm
        mass = xboa.common.pdg_pid_to_mass[13]
        if centile == None:
            centile_amplitude = 4
        else:
            centile_amplitude = scipy.stats.chi2.ppf(centile, 4)
        for data in self.co_data:
            try:
                tm = DecoupledTransferMatrix(self.get_tm(data["tm"]))
                beta = tm.get_beta(0)
            except:
                pass
            p = float(data["variables"]["__momentum__"])*1e3
            beta_rel = p / (p**2+mass**2)**0.5
            eps_eqm_const = 13.6**2/l_r/2/mass/dedx
            eps_eqm = eps_eqm_const*beta/beta_rel
            contour = eps_eqm*centile_amplitude
            x_list.append(data["variables"][x_key])
            y_list.append(contour)
            print(centile, centile_amplitude, eps_eqm, contour)
        label = "4 $\\times \\varepsilon_{eqm}$"
        if not centile is None:
            label = f"{centile*100} centile"
        axes.plot(x_list, y_list, "o-", label = label)

    def plot_a4d_ratio(self, variables, a_4d_good, a_4d_all):
        max_a4d = max([max(a_4d) for a_4d in a_4d_all if len(a_4d)])
        max_a4d = self.a_4d_max #min(max_a4d, self.a_4d_max)
        n_bins = self.n_a4d_bins
        y_bins = [max_a4d/n_bins*i for i in range(n_bins+1)]
        if len(variables[0]) != 1:
            print(variables[0])
            raise ValueError("Confused. "+str(variables[0].keys()))
        var_key = sorted(variables[0].keys())[0]
        a_variable = [var[var_key] for var in variables]
        x_bins = [(a_variable[i]+a_variable[i+1])/2 for i in range(len(a_variable)-1)]
        x_bins.append((a_variable[-1]-a_variable[-2])/2+a_variable[-1])
        x_bins.insert(0, a_variable[0]-(a_variable[1]-a_variable[0])/2)


        x_value, y_value, weight = [], [], []
        for i, var in enumerate(a_variable):
            good_hist = numpy.histogram(a_4d_good[i], y_bins)[0]
            all_hist = numpy.histogram(a_4d_all[i], y_bins)[0]
            print("    with" , str(variables[i]).ljust(20), "got good tracks/bin:", good_hist, sum(good_hist), "of", sum(all_hist), "tracks")
            print("                         all tracks/bin" , all_hist)
            for j in range(n_bins):
                x_value.append(var)
                y_value.append(max_a4d*(j+0.5)/n_bins)
                if all_hist[j]:
                    weight.append(good_hist[j]/all_hist[j])
                else:
                    weight.append(1.0)
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        axes = figure.add_subplot(1, 1, 1)
        hist = axes.hist2d(x_value, y_value, [x_bins, y_bins], weights=weight, label="survival probability")
        axes.set_xlabel(self.key_subs[var_key]+" "+self.units_subs[var_key])
        axes.set_ylabel("A$_{4d}$ [mm]")
        axes.set_title("Survival probability\n"+self.get_title())
        utilities.setup_large_figure(axes)
        for key in self.variables_of_interest:
            if key == var_key:
                continue
                var = [data["substitutions"][key] for data in self.co_data]
        self.plot_eps_eqm(axes, 0.99)
        self.plot_eps_eqm(axes, None)

        beta_axes = axes.twinx()
        self.plot_beta(beta_axes)

        cb = figure.colorbar(hist[3], pad=0.1)
        cb.ax.tick_params(labelsize=14)
        cb.set_label("Survival Probability", fontsize=20)
        #cb.tick_params(labelsize=14)
        axes.legend(loc="upper left")
        beta_axes.legend(loc="center left")
        figure.savefig(os.path.join(self.plot_dir, "amplitude_ratio_"+self.short_form[var_key]+".png"))

    short_form = {
        "__cell_length__":"l",
        "__coil_radius__":"r0",
        "__dipole_field__":"by",
        "__momentum__":"pz",
        "__energy__":None,
    }
    key_subs = {
        "__cell_length__":"Cell Length",
        "__coil_radius__":"Inner Radius",
        "__dipole_field__":"B$_{y}$",
        "__momentum__":"p$_{tot}$",
        "__energy__":None,
    }
    units_subs = {
        "__cell_length__":"[mm]",
        "__coil_radius__":"[mm]",
        "__dipole_field__":"[T]",
        "__momentum__":"[GeV/c]",
        "__energy__":"",
    }
    beta_limit = 1e4
    default_tm = [[ 1.70949964e-01,  1.05529741e+02, -9.35191978e-06, -5.77306748e-03],
                [-9.19815829e-03,  1.71516456e-01,  5.03190738e-07, -9.38291007e-06],
                [ 9.35191978e-06,  5.77306748e-03,  1.70949964e-01,  1.05529741e+02],
                [-5.03190738e-07,  9.38291007e-06, -9.19815829e-03,  1.71516456e-01]]



def fname(prefix, params, ignore_globs):
    fname = f"{prefix}"
    for key, value in params:
        print(key, value)
        if ignore_globs and ("*" in value or "?" in value):
            continue
        fname += f"{key}={value};_"
    return fname[:-2]

def do_plot(run_dir, args):
    run_dir_glob = fname(run_dir, args, False)
    title = run_dir_glob.replace(";_", " ")
    title = title.split("/")[-1]
    plot_dir = fname(run_dir+"/plot_amplitude_2_", args, True)
    file_name = "track_beam_amplitude/da_scan//output*.txt"
    co_file_name = "closed_orbits_cache"
    cell_length = 1600.0 # full cell length
    if "cell_length" in args:
        cell_length = float(args["cell_length"])
    file_format = "icool_for009"
    plotter = PlotG4BL(run_dir_glob, co_file_name, cell_length, file_name, file_format, plot_dir, 40.0)
    plotter.title = title
    plotter.beta_limit = 1e4
    plotter.variables_of_interest = ["__momentum__"]
    plotter.a_4d_max = 160
    plotter.n_a4d_bins = 20
    plotter.do_plots(1, 20)

def main():
    run_dir = "output/demo_oct24_v2/"
    cell_length = 2000
    args = [
        ("pz_beam", "*"),
        ("harmonics", "(7.5)"), # , 3.0, 2.7
        ("cell_length", "2000"),
    ]
    do_plot(run_dir, args)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")

"""
8.8388 0.0
8.8164 0.6297
8.75 1.25
8.6426 1.852
8.4988 2.4282
"""