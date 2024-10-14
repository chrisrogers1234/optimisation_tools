import subprocess
import math
import json
import glob
import operator
import os
import sys

import scipy
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
    def __init__(self, run_dir_glob, file_name_glob, file_format, plot_dir):
        self.plot_dir_root = plot_dir
        self.short_keys_of_interest = []
        self.data = []
        self.px_range = [-150.0, 150.0]
        self.x_range = [-500.0, 500.0]
        self.pz_range = [1.0, 600.0]
        self.e_range = [0.0, 600.0]
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

    def plot_name(self, a_file):
        fname = a_file.split("/")[2]
        fname = fname.split(";_")
        plot_name = ""
        reverse_dict = {v:k for k, v in self.short_form.items()}
        line_length = 0
        for item in fname:
            try:
                short_key, value = item.split("=")
                if short_key not in self.short_keys_of_interest:
                    continue
                long_key = reverse_dict[short_key]
                delta = self.key_subs[long_key]+" "+value+" "+self.units_subs[long_key]+"; "
                plot_name += delta
                line_length += len(delta)
                if line_length > 60:
                    plot_name += "\n"
                    line_length = 0

            except ValueError:
                continue
        plot_name = plot_name[:-2]
        return plot_name

    def glob_data(self):
        print("Globbing", self.run_dir_glob)
        glob_list = sorted(glob.glob(self.run_dir_glob))
        for a_dir in glob_list:
            print("... searching", a_dir)
            for a_file in glob.glob(os.path.join(a_dir, self.file_name_glob)):
                my_data = {"plot_name":self.plot_name(a_file), "file_name":a_file, "bunch_list":[], "plot_dir":os.path.join(a_dir, self.plot_dir_root)}
                my_data["substitutions"] = json.load(open(os.path.join(a_dir, "subs_list.json")))
                self.data.append(my_data)
        if len(glob_list) == 0:
            ls_target = os.path.split(self.run_dir_glob[:-1])[0]
            dir_ls = os.listdir(ls_target)
            for fname in sorted(dir_ls):
                print(fname)
        self.colors =  ["C"+str(i) for i in range(len(self.data))]

    def get_tm(self, tracking_matrix):
        tm = numpy.array([row[1:5] for row in tracking_matrix[0:4]])
        tm = DecoupledTransferMatrix.simplectify(tm)
        return tm

    def do_plots(self):
        self.plot_dir = None
        for i, data in enumerate(self.data):
            if self.analysis_list and i not in self.analysis_list:
                continue
            data["bunch_list"] = Bunch.new_list_from_read_builtin(self.file_format, data["file_name"])
            print(f"Loaded {len(data['bunch_list'])} bunches")
            print("Plots for file", i+1)
            self.plot_dir = data["plot_dir"]
            self.movie_dir = data["plot_dir"]+"/movie"
            utilities.clear_dir(self.plot_dir)
            utilities.clear_dir(self.movie_dir)
            try:
                self.plot_transverse(data)
            except Exception:
                sys.excepthook(*sys.exc_info())
            data["bunch_list"] = []
            print(f"Put plots in\n  {self.plot_dir}\n")

    def exponential_fit_func(self, xdata, a0, a1, a2):
        try:
            return [a2-a0*math.exp(-x/a1) for x in xdata]
        except Exception:
            return [0.0 for x in xdata]

    def get_eps_long(self, bunch):
        my_vars = bunch.list_get_hit_variable(["ct", "energy", "weight"], ["mm", "MeV", ""])
        cov = numpy.cov([my_vars[0], my_vars[1]], aweights=my_vars[2])
        det = numpy.linalg.det(cov)**0.5
        eps = det/xboa.common.pdg_pid_to_mass[13]
        return eps

    def get_eps_6d(self, bunch):
        my_vars = bunch.list_get_hit_variable(
            ["t", "energy", "x", "px", "y", "py", "weight"],
            ["ns", "MeV", "mm", "MeV", "mm", "MeV", ""])
        cov = numpy.cov([my_vars[i] for i in range(6)], aweights=my_vars[6])
        det = numpy.linalg.det(cov)**0.5
        eps = det/xboa.common.pdg_pid_to_mass[13]**3
        return eps

    def rebunch(self, data):
        if not self.frequency or self.frequency <= 0.0:
            return
        rf_t = 1/self.frequency
        self.ct_range = [i*0.6*rf_t*xboa.common.constants["c_light"] for i in [-1, +1]]
        self.t_range = [i*0.6*rf_t for i in [-1, +1]]
        for bunch in data["bunch_list"]:
            t0 = bunch[0]["t"]
            for hit in bunch:
                hit["t"] -= t0
                while hit["t"] > rf_t/2:
                    hit["t"] -= rf_t
                while hit["t"] < -rf_t/2:
                    hit["t"] += rf_t
        print(f"Rebunched with frequency {self.frequency} and time period {rf_t:.2g}")

    def cuts(self, data):
        bunch_list = []
        for bunch in data["bunch_list"]:
            if self.z_range[0] is not None and bunch[0]["z"] < self.z_range[0]:
                continue
            if self.z_range[1] is not None and bunch[0]["z"] > self.z_range[1]:
                continue
            bunch_list.append(bunch)
        data["bunch_list"] = bunch_list

        Bunch.clear_global_weights()
        data["bunch_list"][0].cut({"energy":self.e_min}, operator.lt, global_cut=True)
        data["bunch_list"][0].cut({"energy":self.e_max}, operator.gt, global_cut=True)
        for bunch in data["bunch_list"]:
            bunch.conditional_remove({"pid":-13}, operator.ne)
            ev_repeats = [hit["event_number"] for hit in bunch]
            ev_repeats = set([ev for ev in ev_repeats if ev_repeats.count(ev)-1])
            for ev in ev_repeats:
                bunch.conditional_remove({"event_number":ev}, operator.eq)
        transmission = [bunch.bunch_weight() for bunch in data["bunch_list"]]
        for bunch in data["bunch_list"]:
            bunch.transmission_cut(data["bunch_list"][-1], global_cut=True)
            bunch_weight, new_bunch_weight = -1, -2
            while bunch_weight != new_bunch_weight:
                eps = bunch.get_emittance(['x', 'y'])
                try:
                    pass #bunch.cut({"amplitude x y":eps*20}, operator.gt, global_cut=True)
                except numpy.linalg.LinAlgError:
                    break
                bunch_weight = new_bunch_weight
                new_bunch_weight = bunch.bunch_weight()
        return transmission

    def scraping_plot(self, bunch, plot_name):
        figure = matplotlib.pyplot.figure()
        figure.suptitle(plot_name)
        axes = figure.add_subplot(1, 1, 1)
        axes.set_xlabel("$A_\\perp$ [mm]")
        axes.set_ylabel("$A_{//}$ [mm]")
        x_range = [0, 50]
        y_range = [0, 50]
        n_bins = [20, 20]
        my_vars = bunch.list_get_hit_variable(["amplitude x y", "amplitude ct", "weight"], ["mm", "mm", ""])
        amplitude_perp = [a for i, a in enumerate(my_vars[0]) if my_vars[2][i] < 0.1]
        amplitude_long = [a for i, a in enumerate(my_vars[1]) if my_vars[2][i] < 0.1]
        h_all, xedges, yedges, image = axes.hist2d(my_vars[0], my_vars[1], bins=n_bins, range=[x_range, y_range])
        figure.savefig(self.plot_dir+"/initial_beam_amplitude.png")
        axes.clear()
        axes.set_xlabel("$A_\\perp$ [mm]")
        axes.set_ylabel("$A_{//}$ [mm]")
        h_scraped, xedges, yedges, image = axes.hist2d(amplitude_perp, amplitude_long, bins=n_bins, range=[x_range, y_range])
        figure.savefig(self.plot_dir+"/scraped_particles_amplitude.png")

        x_points = []
        y_points = []
        h_ratio = []
        for ix in range(n_bins[0]):
            for iy in range(n_bins[1]):
                x_points.append((xedges[ix]+xedges[ix+1])/2)
                y_points.append((yedges[iy]+yedges[iy+1])/2)
                if h_all[ix][iy] == 0:
                    h_ratio.append(0.0)
                else:
                    h_ratio.append(h_scraped[ix][iy]/h_all[ix][iy])
        axes.clear()
        axes.set_xlabel("$A_\\perp$ [mm]")
        axes.set_ylabel("$A_{//}$ [mm]")
        ahist, xedges, yedges, image = axes.hist2d(x_points, y_points, weights=h_ratio, bins=n_bins, range=[x_range, y_range])
        figure.colorbar(image)
        figure.savefig(self.plot_dir+"/scraping_ratio_amplitude.png")

    def emittance_plot(self, data):
        z_list = self.get_z_list()
        figure = matplotlib.pyplot.figure()
        figure.suptitle(data["plot_name"])
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(z_list, self.out_json["emittance_perp"], label="$\\varepsilon_{\\perp}$")
        axes.set_xlabel("z [m]")
        axes.set_ylabel("$\\varepsilon_{\\perp}, \\varepsilon_{//}, $ [mm]")
        print("Fitting")
        eps_str = "$\\varepsilon^{eqm,1} exp(z/z_0)-\\varepsilon^{eqm,2}$"
        try:
            parameters, errors = scipy.optimize.curve_fit(self.exponential_fit_func, z_list, self.out_json["emittance_perp"], p0=[-2, 50.0, +2])
            emittance_fit = self.exponential_fit_func(z_list, *parameters)
        except RuntimeError:
            emittance_fit = [0 for i in z]
            parameters = [-99, -99, -99]
        self.out_json["emittance_perp_fit"] = parameters.tolist()
        eps_str = "$\\varepsilon^{eqm,1}_{\\perp}=$"+str(round(parameters[0], 3))+" mm"
        eps_str += "; $\\varepsilon^{eqm,2}_{\\perp}=$"+str(round(parameters[2], 3))+" mm"
        eps_str += "; $z_0=$"+str(round(parameters[1]))+" m"
        #axes.text(0.4, 0.84, eps_str, transform=axes.transAxes, color="blue")
        axes.plot(z_list, emittance_fit, linestyle="dashed", color="lightblue")
        axes.plot(z_list, self.out_json["emittance_long"], c="r", label="$\\varepsilon_{//}$")
        lim = axes.get_ylim()
        axes.set_ylim([0.0, lim[1]])
        try:
            parameters, errors = scipy.optimize.curve_fit(self.exponential_fit_func, z_list, self.out_json["emittance_long"], p0=[-2, 50.0, +2])
            emittance_fit = self.exponential_fit_func(z_list, *parameters)
            axes.plot(z_list, emittance_fit, linestyle="dashed", color="pink")
        except RuntimeError:
            emittance_fit = [0 for i in z]
            parameters = [-99, -99, -99]

        axes = axes.twinx()
        axes.plot(z_list, self.out_json["emittance_6d"], c="black", label="$\\varepsilon_{6d}$")
        axes.set_ylabel("$\\varepsilon_{6d}$ [mm$^3$]")
        lim = axes.get_ylim()
        axes.set_ylim([0.0, lim[1]])
        eps_str = "$\\varepsilon^{eqm,1}_{//}=$"+str(round(parameters[0], 3))+" mm"
        eps_str += "; $\\varepsilon^{eqm,2}_{//}=$"+str(round(parameters[2], 3))+" mm"
        eps_str += "; $z_0=$"+str(round(parameters[1]))+" m"
        self.out_json["emittance_long_fit"] = parameters.tolist()
        #trans_str = "Transmission: "+str(round(data["bunch_list"][0].bunch_weight()/len(data["bunch_list"][0])*100, 1))+" %"
        #axes.text(0.4, 0.68, trans_str, transform=axes.transAxes)
        figure.legend()
        figure.savefig(self.plot_dir+"/emittance_vs_z.png")

    def get_z_list(self):
        z_list = [(z - self.out_json["z"][0])/xboa.common.units["m"] for z in self.out_json["z"]]
        return z_list

    def get(self, var, axes, bunch):
        try:
            b = bunch.get(var, axes)
        except ZeroDivisionError:
            b = 0.0
        if math.isnan(b) or math.isinf(b):
            b = 0.0
        return b

    def plots_beamline(self, data, transmission):
        self.out_json = {}
        self.out_json["z"] = [bunch[0]["z"] for bunch in data["bunch_list"]]
        self.out_json["weights"] = [bunch.bunch_weight() for bunch in data["bunch_list"]]
        self.out_json["transmission"] = transmission
        self.out_json["percent_transmission"] = [100.0*t/transmission[0] for t in transmission]
        self.out_json["emittance_perp"] = [self.get("emittance", ["x", "y"], bunch) for bunch in data["bunch_list"]]
        self.out_json["emittance_long"] = [self.get_eps_long(bunch) for bunch in data["bunch_list"]]
        self.out_json["emittance_6d"] = [self.get_eps_6d(bunch) for bunch in data["bunch_list"]]
        self.out_json["ref_momentum"] = [bunch[0]["p"] for bunch in data["bunch_list"]]
        self.out_json["mean_momentum"] = [self.get("mean", ["p"], bunch) for bunch in data["bunch_list"]]
        self.out_json["beta"] = [self.get("beta", ["x", "y"], bunch) for bunch in data["bunch_list"]]
        for key in self.out_json:
            self.out_json[key] = [float(x) for x in self.out_json[key] if x == x]
        print("  z [m]       ", "".join([f"{z/1e3:8.4g}" for z in self.out_json["z"][::10]]))
        print("  Transmission", "".join([f"{trans:8.4g}" for trans in self.out_json["percent_transmission"][::10]]))
        print("  Trans emit  ", "".join([f"{emit:8.4g}" for emit in self.out_json["emittance_perp"] [::10]]))
        print("  Long emit   ", "".join([f"{emit:8.4g}" for emit in self.out_json["emittance_long"] [::10]]))

        z_list = self.get_z_list()
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        figure.suptitle(data["plot_name"])
        axes = figure.add_subplot(1, 2, 1)
        axes.plot(z_list, self.out_json["beta"])
        axes.set_xlabel("z [m]")
        axes.set_ylabel("$\\beta_\\perp$ [mm]")


        axes = figure.add_subplot(1, 2, 2)
        axes.plot(z_list, self.out_json["mean_momentum"], label="Mean p$")
        axes.plot(z_list, self.out_json["ref_momentum"], c="g", linestyle="dashed", label="ref p$")
        axes.set_xlabel("z [m]")
        axes.set_ylabel("p [MeV/c]")
        figure.savefig(self.plot_dir+"/optics_vs_z.png")

        self.transmission_plot(data)
        self.emittance_plot(data)
        #self.scraping_plot(data["bunch_list"][0], data["plot_name"])

        self.out_json["substitutions"] = data["substitutions"]
        self.out_json["file_name"] = data["file_name"]
        with open(self.plot_dir+"/"+self.output_filename, "w") as fout:
            json.dump(self.out_json, fout)

    def transmission_plot(self, data):
        z_list = self.get_z_list()
        figure = matplotlib.pyplot.figure()
        figure.suptitle(data["plot_name"])
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(z_list, self.out_json["percent_transmission"])
        axes.set_ylim([0.0, 110.0])
        axes.set_xlabel("z [m]")
        axes.set_ylabel("Transmission [%]")
        figure.savefig(self.plot_dir+"/transmission_vs_z.png")

    def plot_transverse(self, data):
        self.rebunch(data)
        transmission = self.cuts(data)
        self.plots_beamline(data, transmission)
        if self.max_station < 1:
            bunch_list = [data["bunch_list"][0]+data["bunch_list"][-1]]
        else:
            bunch_list = data["bunch_list"][:self.max_station:self.station_stroke]

        for frame_index, bunch in enumerate(bunch_list):
            psize = 3
            print("\r    Making movie frame", frame_index+1, "/", len(bunch_list), end="")
            my_vars = bunch.list_get_hit_variable(["x", "px", "y", "py", "t", "energy", "weight"], ["mm", "mm", "MeV/c", "MeV/c", "ns", "MeV", ""])
            bunch.conditional_remove({"weight":0.00001}, operator.lt)
            my_vars_cut = bunch.list_get_hit_variable(["x", "px", "y", "py", "t", "energy", "weight"], ["mm", "mm", "MeV/c", "MeV/c", "ns", "MeV", ""])

            figure = matplotlib.pyplot.figure(figsize=(20,10))
            figure.suptitle(f"{data['plot_name']}\nz: {bunch[0]['z']/xboa.common.units['m']} m; N: {len(my_vars_cut[0])}/{len(my_vars[0])}")
            axes = figure.add_subplot(2, 2, 1)
            axes.scatter(my_vars[0], my_vars[1], c="orange", s=psize)
            axes.scatter(my_vars_cut[0], my_vars_cut[1], s=psize)
            axes.set_xlabel("x [mm]")
            axes.set_ylabel("p$_{x}$ [MeV/c]")
            axes.set_xlim(self.x_range)
            axes.set_ylim(self.px_range)
            eps = bunch.get("emittance", ["x", "y"])
            test_str = "$\\sigma$(x) "+str(round(numpy.std(my_vars[0]), 1))+" mm\n"
            test_str += "$\\sigma$(p$_x$) "+str(round(numpy.std(my_vars[1]), 1))+" MeV/c\n"
            test_str += "$\\sigma$(y) "+str(round(numpy.std(my_vars[2]), 1))+" mm\n"
            test_str += "$\\sigma$(p$_y$) "+str(round(numpy.std(my_vars[3]), 1))+" MeV/c\n"
            test_str += "$\\varepsilon_{x}$ "+str(round(bunch.get("emittance", ["x"]), 1))+" mm\n"
            test_str += "$\\varepsilon_{y}$ "+str(round(bunch.get("emittance", ["y"]), 1))+" mm\n"
            test_str += "$\\varepsilon_{\\perp}$ "+str(round(eps, 1))+" mm"
            axes.text(0.01, 0.7, test_str, transform=axes.transAxes)

            my_vars[4] = [ct-my_vars[4][0] for ct in my_vars[4]]
            my_vars_cut[4] = [ct-my_vars_cut[4][0] for ct in my_vars_cut[4]]
            axes = figure.add_subplot(2, 2, 2)
            axes.scatter(my_vars[4], my_vars[5], c="orange", s=psize)
            axes.scatter(my_vars_cut[4], my_vars_cut[5], s=psize)
            axes.set_xlabel("t [ns]")
            axes.set_ylabel("Total energy [MeV]")
            axes.set_xlim(self.t_range)
            axes.set_ylim(self.e_range)
            eps = self.get_eps_long(bunch)
            test_str = "$\\sigma$(t) "+str(round(numpy.std(my_vars[4]), 1))+" ns\n"
            test_str += "$\\sigma$(E) "+str(round(numpy.std(my_vars[5]), 1))+" MeV\n"
            test_str += "$\\varepsilon_{//}$ "+str(round(eps, 1))+" mm"
            axes.text(0.01, 0.8, test_str, transform=axes.transAxes)

            axes = figure.add_subplot(2, 2, 3)
            axes.scatter(my_vars[0], my_vars[5], c="orange", s=psize)
            axes.scatter(my_vars_cut[0], my_vars_cut[5], s=psize)
            axes.set_xlabel("x [mm]")
            axes.set_ylabel("Total energy [MeV]")
            axes.set_xlim(self.x_range)
            axes.set_ylim(self.e_range)

            axes = figure.add_subplot(2, 2, 4)
            axes.scatter(my_vars[2], my_vars[5], c="orange", s=psize)
            axes.scatter(my_vars_cut[2], my_vars_cut[5], s=psize)
            axes.set_xlabel("y [mm]")
            axes.set_ylabel("Total energy [MeV]")
            axes.set_xlim(self.x_range)
            axes.set_ylim(self.e_range)

            station = str(bunch[0]["station"]).rjust(4, "0")
            figure.savefig(self.movie_dir+"/ps_"+station+".png")
            matplotlib.pyplot.close(figure)
        #mencoder mf://turn*.png -mf w=800:h=600:fps=5:type=png -ovc lavc -lavcopts vcodec=msmpeg4:mbd=2:trell -oac copy -o injection.avi

        print("    mencoding")
        here = os.getcwd()
        os.chdir(self.movie_dir)
        try:
            output = subprocess.check_output(["mencoder",
                                    "mf://ps_*.png",
                                    "-mf", "w=800:h=600:fps=5:type=png",
                                    "-ovc", "lavc",
                                    "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                    "-oac", "copy",
                                    "-o", "movie.avi"], stderr=subprocess.STDOUT)
        except Exception:
            sys.excepthook(*sys.exc_info())
            print("Movie failed with:")
            print(output)
        os.chdir(here)

    @classmethod
    def get_subs(cls, item):
        return item["substitutions"][0]
        if "file_name" not in item:
            print("Failed", item["substitutions"])
        subs_dir = os.path.split(item["file_name"])[0]
        subs_name = os.path.join(subs_dir, "subs.json")
        subs_str = open(subs_name).read()
        subs_json = json.loads(subs_str)
        return subs_json

    @classmethod
    def plot_scan(cls, glob_name, x_parameter, y_lambda, y_label, color_parameter=None):
        glob_list = sorted(glob.glob(glob_name))
        if len(glob_list) == 0:
            raise RuntimeError(f"Failed to find any files matching {glob_name}")
        #print("Plot scan loading")
        #for file_name in glob_list:
        #    print(f"    {file_name}")
        json_list = [json.load(open(file_name)) for file_name in glob_list]
        x_list = [cls.get_subs(item)[x_parameter] for item in json_list]
        y_list = [y_lambda(item) for item in json_list]
        print("   Plotting ", len(json_list), "points for", y_label)
        print("     ", x_parameter, x_list)
        print("     ", y_label, y_list)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        if color_parameter == None:
            axes.scatter(x_list, y_list)
        else:
            c_list = [cls.get_subs(item)[color_parameter] for item in json_list]
            print("   ", color_parameter, c_list)
            scatter = axes.scatter(x_list, y_list, c=c_list)
            c_label = f"{cls.key_subs[color_parameter]} {cls.units_subs[color_parameter]}"
            axes.text(1.0, 1.03, c_label, transform=axes.transAxes)
            figure.colorbar(scatter)
        axes.set_ylabel(y_label)
        axes.set_xlabel(cls.key_subs[x_parameter]+" "+cls.units_subs[x_parameter])
        return figure

    @classmethod
    def plot_scan_group(self, glob_name, parameters_of_interest, scan_plot_dir, plot_title=""):
        start_cell = self.start_cell
        mid_cell = self.mid_cell
        target_cell = self.target_cell
        y_lambda_list = [
            lambda item: item["emittance_long"][start_cell],
            lambda item: item["emittance_perp"][start_cell],
            lambda item: item["emittance_long"][mid_cell],
            lambda item: item["emittance_perp"][mid_cell],
            lambda item: item["emittance_long"][target_cell],
            lambda item: item["emittance_perp"][target_cell],
            lambda item: item["emittance_long_fit"][2],
            lambda item: item["emittance_perp_fit"][2],
            lambda item: item["transmission"][target_cell]/item["transmission"][start_cell]*100,
            lambda item: item["transmission"][target_cell]/item["transmission"][mid_cell]*100,
        ]
        y_label_list = [
            f"$\\varepsilon_{{//}}$ at cell {start_cell} [mm]",
            f"$\\varepsilon_\\perp$ at cell {start_cell} [mm]",
            f"$\\varepsilon_{{//}}$ at cell {mid_cell} [mm]",
            f"$\\varepsilon_\\perp$ at cell {mid_cell} [mm]",
            f"$\\varepsilon_{{//}}$ at cell {target_cell} [mm]",
            f"$\\varepsilon_\\perp$ at cell {target_cell} [mm]",
            "Estimated $\\varepsilon^{eqm}_{//}$ [mm]",
            "Estimated $\\varepsilon^{eqm}_\\perp$ [mm]",
            f"Transmission from cell {start_cell} to {target_cell} [%]",
            f"Transmission from cell {mid_cell} to {target_cell} [%]",
        ]
        y_fname_list = [
            f"eps_long_cell_{start_cell}",
            f"eps_trans_cell_{start_cell}",
            f"eps_long_cell_{mid_cell}",
            f"eps_trans_cell_{mid_cell}",
            f"eps_long_cell_{target_cell}",
            f"eps_trans_cell_{target_cell}",
            "eps_long_eqm",
            "eps_trans_eqm",
            f"transmission_{start_cell}_{target_cell}",
            f"transmission_{mid_cell}_{target_cell}"
        ]
        y_range_list = [
            [0, 200],
            [0, 25],
            [0, 200],
            [0, 25],
            [0, 200],
            [0, 25],
            [0, 200],
            [0, 25],
            [40, 80],
            [80, 100],
        ]
        utilities.clear_dir(scan_plot_dir)
        for i, x_parameter in enumerate(parameters_of_interest):
            c_parameter = None
            if len(parameters_of_interest) > 1:
                c_parameter_list = parameters_of_interest[1:]+[parameters_of_interest[0]]
                c_parameter = c_parameter_list[i]
                print("Plotting x: ", x_parameter, "c:", c_parameter)
            for jy, y_lambda in enumerate(y_lambda_list):
                y_label = y_label_list[jy]
                figure = self.plot_scan(glob_name, x_parameter, y_lambda, y_label, c_parameter)
                figure.axes[0].set_ylim(y_range_list[jy])
                figure.axes[0].grid(True, axis="y")
                if plot_title:
                    figure.suptitle(plot_title)
                xname = x_parameter.replace("__", "")
                fname = f"{scan_plot_dir}/{y_fname_list[jy]}-{xname}.png"
                figure.savefig(fname)
                print("    Saved", fname)
            print("\n")

    start_cell = 0
    mid_cell = 20
    target_cell = 25



    short_form = {
        "__dipole_field__":"by",
        "__momentum__":"pz_beam",
        "__coil_radius__":"r0",
        "__energy__":None,
        "__wedge_angle__":"wedge_angle",
        "__dipole_polarity1__":"dp",
        "__wedge_thickness__":"wedge_thickness",
        "__version__":"version",
        "__polarity__":"polarity",
        "__rf_efield__":"rf_efield",
        "__rf_phase__":"rf_phase",
        "__rf_window_thickness__":"rf_wt",
        "__eps_t__":"eps_t",
        "__eps_l__":"eps_l",
        "__material__":"material",
        "__rf_iris_factor__":"iris_factor",
        "__harmonic_0__":"harmonic_0",
        "__n_rf__":"n_rf",
        "__cell_length__":"cell_length",
    }
    key_subs = {
            "__wedge_angle__":"$\\theta_{wdg}$",
            "__coil_radius__":"r$_{coil}$",
            "__dipole_field__":"B$_{y}$",
            "__momentum__":"p$_{tot}$",
            "__energy__":None,
            "__material__":"",
            "__dipole_polarity1__":"Dipole Polarity",
            "__wedge_thickness__":"$L_{wdg}$",
            "__version__":"",
            "__polarity__":"",
            "__rf_efield__":"$E_0$",
            "__rf_phase__":"$\\phi_s$",
            "__eps_t__":"$\\varepsilon_{\\perp}$",
            "__eps_l__":"$\\varepsilon_{//}$",
            "__rf_window_thickness__":"window thickness",
            "__rf_iris_factor__":"Iris Factor",
            "__harmonic_0__":"$B_{0}$",
            "__n_rf__":"N(rf)",
            "__cell_length__":"Cell Length",
    }
    units_subs = {
            "__coil_radius__":"[mm]",
            "__wedge_angle__":"[deg]",
            "__dipole_field__":"[T]",
            "__momentum__":"[MeV/c]",
            "__energy__":"",
            "__dipole_polarity1__":"",
            "__wedge_thickness__":"[mm]",
            "__version__":"",
            "__polarity__":"",
            "__rf_efield__":"[MV/m]",
            "__rf_phase__":"[deg]",
            "__eps_t__":"mm",
            "__eps_l__":"meV s",
            "__material__":"",
            "__rf_window_thickness__":"[mm]",
            "__rf_iris_factor__":"",
            "__harmonic_0__":"[T]",
            "__n_rf__":"",
            "__cell_length__":"[mm]",

    }
    beta_limit = 4e3
    el_limit = 25

def make_run_dir_glob(run_dir, variables):
    a_glob = run_dir+"/"
    for key, value in variables:
        a_glob += key+"="+value+";_"
    a_glob = a_glob[:-2]+"/"
    return a_glob

def main():
    tracking = "beam_file"
    variables = [
        ("pz_beam", "300.0"),
        ("harmonic_0", "*"),
        ("iris_factor", "0.5"),
        ("n_rf", "*"),
        ("by", "0.6"),
        ("polarity", "++++"),
        ("wedge_thickness", "*"),
        ("wedge_angle", "*"),
        ("rf_efield", "25"),
        ("rf_phase", "*"),
        ("cell_length", "*"),
    ]
    run_dir = "output/rectilinear_v35"
    run_dir_glob = make_run_dir_glob(run_dir, variables)
    plot_dir = f"emittance_plots_{tracking}/"
    file_name = f"track_beam_amplitude/{tracking}/output.txt"
    file_format = "icool_for009"
    frequency = 0.176

    #run_dir = "output/rectilinear_prab/simulation_v1/stage_a1-no-decay"
    #run_dir_glob = run_dir
    #file_name = "particles_info.txt"
    #frequency = 0.352

    plotter = PlotG4BL(run_dir_glob, file_name, file_format, plot_dir)
    plotter.analysis_list = None
    for key, value in variables:
        if "*" in value or "?" in value:
            plotter.short_keys_of_interest.append(key)
    plotter.short_keys_of_interest = ["by", "wedge_thickness", "wedge_angle", "rf_efield"]
    plotter.e_min = 0.0
    plotter.e_max = 500.0
    plotter.z_range = [0*1e3, 2000*1e3] # [None, None]#
    plotter.max_station = 1000 # max station for phase space plots
    plotter.station_stroke = 50 # stroke station for phase space plots
    plotter.frequency = frequency
    plotter.glob_data()
    #plotter.do_plots()
    index = [var[0] for var in variables].index("rf_efield")
    efield_for_scan = variables[index][1]
    index = [var[0] for var in variables].index("by")
    by_for_scan = variables[index][1]
    scan_glob = run_dir_glob+plot_dir+"/"+plotter.output_filename
    scan_plot_dir = run_dir+f"/scan_plots_{efield_for_scan}MVm"
    for key in plotter.short_keys_of_interest:
            scan_plot_dir += f"_{key}"
    PlotG4BL.start_cell = 0
    PlotG4BL.mid_cell = 25
    PlotG4BL.target_cell = 50
    plotter.plot_scan_group(scan_glob, ["__wedge_angle__", "__wedge_thickness__"], scan_plot_dir, f"E$_0$ = {efield_for_scan} MV/m; $B_y$ = {by_for_scan} T")



if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
