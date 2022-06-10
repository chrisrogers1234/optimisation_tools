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
        self.data = []
        self.variables_of_interest = None
        self.px_range = [-80.0, 80.0]
        self.x_range = [-100.0, 100.0]
        self.pz_range = [150.0, 250.0]
        self.e_range = [150.0, 325.0]
        self.ct_range = [-300.0, 300.0]
        self.z_range = [None, None]
        self.file_format = file_format
        self.glob_data(run_dir_glob, file_name_glob)    
        self.colors =  ["C"+str(i) for i in range(len(self.data))]
        self.title = ""
        self.max_station = 10000
        self.analysis_list = None

    def plot_name(self, a_file):
        fname = a_file.split("/")[2]
        fname = fname.split("_")
        plot_name = ""
        reverse_dict = {v:k for k, v in self.short_form.items()}
        for item in fname:
            try:
                short_key, value = item.split("=")
                long_key = reverse_dict[short_key]
                plot_name += self.key_subs[long_key]+" "+value+" "+self.units_subs[long_key]+"; "
            except ValueError:
                continue
        plot_name = plot_name[:-2]
        return plot_name

    def glob_data(self, run_dir_glob, file_name_glob):
        print("Globbing", run_dir_glob)
        for a_dir in sorted(glob.glob(run_dir_glob)):
            print("... searching", a_dir)
            for a_file in glob.glob(os.path.join(a_dir, file_name_glob)):
                my_data = {"plot_name":self.plot_name(a_file), "file_name":a_file, "bunch_list":[], "plot_dir":os.path.join(a_dir, self.plot_dir_root)}
                self.data.append(my_data)

    def get_tm(self, tracking_matrix):
        tm = numpy.array([row[1:5] for row in tracking_matrix[0:4]])
        tm = DecoupledTransferMatrix.simplectify(tm)
        return tm

    def do_plots(self):
        #self.title = self.get_title()
        for i, data in enumerate(self.data):
            if self.analysis_list and i not in self.analysis_list:
                continue
            data["bunch_list"] = Bunch.new_list_from_read_builtin(self.file_format, data["file_name"])
            print("Plots for file", i+1)
            self.plot_dir = data["plot_dir"]
            self.movie_dir = data["plot_dir"]+"/movie"
            utilities.clear_dir(self.plot_dir)
            utilities.clear_dir(self.movie_dir)
            self.plot_transverse(data)
            data["bunch_list"] = []

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
            ["ct", "energy", "x", "px", "y", "py", "weight"],
            ["mm", "MeV", "mm", "MeV", "mm", "MeV", ""])
        cov = numpy.cov([my_vars[i] for i in range(6)], aweights=my_vars[6])
        det = numpy.linalg.det(cov)**0.5
        eps = det/xboa.common.pdg_pid_to_mass[13]**3
        return eps



    def plot_transverse(self, data):
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
            #bunch.conditional_remove({"r":200}, operator.gt)
            ev_repeats = [hit["event_number"] for hit in bunch]
            ev_repeats = set([ev for ev in ev_repeats if ev_repeats.count(ev)-1])
            for ev in ev_repeats:
                bunch.conditional_remove({"event_number":ev}, operator.eq)
        transmission = [bunch.bunch_weight() for bunch in data["bunch_list"]]
        transmission = [100.0*t/transmission[0] for t in transmission]
        for bunch in data["bunch_list"]:
            bunch.transmission_cut(data["bunch_list"][-1], True)
            bunch.conditional_remove({"weight":0.00001}, operator.lt)

        print("Weights (plotted)", [bunch.bunch_weight() for bunch in data["bunch_list"]])
        print("Transmission [%] ", transmission)
        emittance_xy = [bunch.get("emittance", ["x", "y"]) for bunch in data["bunch_list"]]
        emittance_ct = [self.get_eps_long(bunch) for bunch in data["bunch_list"]]
        emittance_6d_alt = [self.get_eps_6d(bunch) for bunch in data["bunch_list"]]
        emittance_6d = [emittance_ct[i]*emittance_xy[i]**2 for i in range(len(emittance_ct))]
        print("Transverse emittance\n", emittance_xy)
        print("Longitudinal emittance\n", emittance_ct)
        print("6D emittance\n", emittance_6d)
        print("6D emittance alt\n", emittance_6d_alt)
        p_ref = [bunch[0]["p"] for bunch in data["bunch_list"]]
        p = [bunch.get("mean", ["p"]) for bunch in data["bunch_list"]]
        beta = []
        for bunch in data["bunch_list"]:
            try:
                beta.append(bunch.get("beta", ["x", "y"]))
                if beta[-1] > self.beta_limit:
                    beta[-1] = 0.0
            except Exception:
                beta.append(0)
        for i, el in enumerate(emittance_ct):
            if el > self.el_limit:
                emittance_ct[i] = 0.0
        z0 = data["bunch_list"][0][0]["z"]
        z = [(bunch[0]["z"]-z0)/xboa.common.units["m"] for bunch in data["bunch_list"]]
        print("z position\n", z)
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        figure.suptitle(data["plot_name"])
        axes = figure.add_subplot(2, 2, 1)
        axes.plot(z, beta)
        axes.set_xlabel("z [m]")
        axes.set_ylabel("$\\beta_\\perp$ [mm]")


        axes = figure.add_subplot(2, 2, 2)
        axes.plot(z, p, label="Mean p$")
        axes.plot(z, p_ref, c="g", linestyle="dashed", label="ref p$")
        axes.set_xlabel("z [m]")
        axes.set_ylabel("p [MeV/c]")

        axes = figure.add_subplot(2, 2, 4)
        axes.plot(z, transmission)
        axes.set_ylim([0.0, 110.0])
        axes.set_xlabel("z [m]")
        axes.set_ylabel("Transmission [%]")
        figure.savefig(self.plot_dir+"/performance_vs_z.png")

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(z, emittance_xy)
        #axes.set_ylim([0.0, axes.get_ylim()[1]])
        axes.set_xlabel("z [m]")
        axes.set_ylabel("$\\varepsilon_{\\perp}, \\varepsilon_{//}, $ [mm]")
        print("Fitting")
        eps_str = "$\\varepsilon^{eqm,1} exp(z/z_0)-\\varepsilon^{eqm,2}$"
        #axes.text(0.4, 0.92, eps_str, transform=axes.transAxes, color="black")

        try:
            emittance_xy = [eps if eps == eps else 0 for eps in emittance_xy]
            parameters, errors = scipy.optimize.curve_fit(self.exponential_fit_func, z, emittance_xy, p0=[-2, 50.0, +2])
            emittance_fit = self.exponential_fit_func(z, *parameters)
        except RuntimeError:
            emittance_fit = [0 for i in z]
            parameters = [-99, -99, -99]
        eps_str = "$\\varepsilon^{eqm,1}_{\\perp}=$"+str(round(parameters[0], 3))+" mm"
        eps_str += "; $\\varepsilon^{eqm,2}_{\\perp}=$"+str(round(parameters[2], 3))+" mm"
        eps_str += "; $z_0=$"+str(round(parameters[1]))+" m"
        #axes.text(0.4, 0.84, eps_str, transform=axes.transAxes, color="blue")
        axes.plot(z, emittance_fit, linestyle="dashed", color="lightblue")
        axes.plot(z, emittance_ct, c="r")


        try:
            emittance_ct = [eps if eps == eps else 0 for eps in emittance_ct]
            parameters, errors = scipy.optimize.curve_fit(self.exponential_fit_func, z, emittance_ct, p0=[-2, 50.0, +2])
            emittance_fit = self.exponential_fit_func(z, *parameters)
            axes.plot(z, emittance_fit, linestyle="dashed", color="pink")
        except RuntimeError:
            emittance_fit = [0 for i in z]
            parameters = [-99, -99, -99]

        axes = axes.twinx()
        axes.plot(z, emittance_6d_alt, c="black")
        axes.set_ylabel("$\\varepsilon_{6d}$ [mm$^3$]")
        #axes.set_ylim([0.0, axes.get_ylim()[1]])
        eps_str = "$\\varepsilon^{eqm,1}_{//}=$"+str(round(parameters[0], 3))+" mm"
        eps_str += "; $\\varepsilon^{eqm,2}_{//}=$"+str(round(parameters[2], 3))+" mm"
        eps_str += "; $z_0=$"+str(round(parameters[1]))+" m"
        #axes.text(0.4, 0.76, eps_str, transform=axes.transAxes, color="red")

        trans_str = "Transmission: "+str(round(data["bunch_list"][0].bunch_weight()/len(data["bunch_list"][0])*100, 1))+" %"
        #axes.text(0.4, 0.68, trans_str, transform=axes.transAxes)

        figure.savefig(self.plot_dir+"/emittance_vs_z.png")
        #matplotlib.pyplot.close(figure)

        print()
        for i, bunch in enumerate(data["bunch_list"][:self.max_station]):
            print("\r    Making movie frames", i, "/", len(data["bunch_list"]), end="")
            my_vars = bunch.list_get_hit_variable(["x", "px", "y", "py", "ct", "energy", "weight"], ["mm", "mm", "MeV/c", "MeV/c", "mm", "MeV", ""])
            bunch.conditional_remove({"weight":0.00001}, operator.lt)
            my_vars_cut = bunch.list_get_hit_variable(["x", "px", "y", "py", "ct", "energy", "weight"], ["mm", "mm", "MeV/c", "MeV/c", "mm", "MeV", ""])

            figure = matplotlib.pyplot.figure(figsize=(20,10))
            figure.suptitle(data["plot_name"]+"\nz: "+str(bunch[0]['z']/xboa.common.units["m"])+" m; N: "+str(len(bunch)))
            axes = figure.add_subplot(2, 2, 1)
            axes.scatter(my_vars[0], my_vars[1], c="orange")
            axes.scatter(my_vars_cut[0], my_vars_cut[1])
            axes.set_xlabel("x [mm]")
            axes.set_ylabel("p$_{x}$ [MeV/c]")
            axes.set_xlim(self.x_range)
            axes.set_ylim(self.px_range)

            my_vars[4] = [ct-my_vars[4][0] for ct in my_vars[4]]
            my_vars_cut[4] = [ct-my_vars_cut[4][0] for ct in my_vars_cut[4]]
            axes = figure.add_subplot(2, 2, 2)
            axes.scatter(my_vars[4], my_vars[5], c="orange")
            axes.scatter(my_vars_cut[4], my_vars_cut[5])
            axes.set_xlabel("ct [mm]")
            axes.set_ylabel("Total energy [MeV]")
            axes.set_xlim(self.ct_range)
            axes.set_ylim(self.e_range)
            eps = self.get_eps_long(bunch)
            test_str = "$\\sigma$(ct) "+str(round(numpy.std(my_vars[4]), 1))+" mm\n"
            test_str += "$\\sigma$(E) "+str(round(numpy.std(my_vars[5]), 1))+" MeV\n"
            test_str += "$\\varepsilon_{//}$ "+str(round(eps, 1))+" mm"
            axes.text(0.01, 0.8, test_str, transform=axes.transAxes)

            axes = figure.add_subplot(2, 2, 3)
            axes.scatter(my_vars[0], my_vars[5], c="orange")
            axes.scatter(my_vars_cut[0], my_vars_cut[5])
            axes.set_xlabel("x [mm]")
            axes.set_ylabel("Total energy [MeV]")
            axes.set_xlim(self.x_range)
            axes.set_ylim(self.e_range)

            axes = figure.add_subplot(2, 2, 4)
            axes.scatter(my_vars[2], my_vars[5], c="orange")
            axes.scatter(my_vars_cut[2], my_vars_cut[5])
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


    def get_title(self, data):
        """
        if len(data) == 0:
            data = self.data
        if len(data) == 0:
            key_list = []
            for key in self.variables_of_interest:
                values = [data["substitutions"][key] for data in self.co_data]
                if numpy.std(values) < 1e-12:
                    key_list.append(key)
                print(key, key_list, numpy.std(values))
            data = self.co_data[0]
        else:
        """
        key_list = self.variables_of_interest

        title = ""
        for key in key_list:
            sub = round(data["substitutions"][key], 3)
            title += self.key_subs[key]+" "+str(sub)+" "+self.units_subs[key]+"; "
        return title[:-2]


    short_form = {
        "__dipole_field__":"by",
        "__momentum__":"pz",
        "__coil_radius__":"r0",
        "__energy__":None,
        "__wedge_opening_angle__":"wq",
        "__dipole_polarity1__":"dp",
    }
    key_subs = {
            "__wedge_opening_angle__":"$\\theta_w$",
            "__coil_radius__":"r$_{coil}$",
            "__dipole_field__":"B$_{y}$",
            "__momentum__":"p$_{tot}$",
            "__energy__":None,
            "__dipole_polarity1__":"Dipole Polarity",
    }
    units_subs = {
            "__coil_radius__":"[mm]",
            "__wedge_opening_angle__":"[deg]",
            "__dipole_field__":"[T]",
            "__momentum__":"[MeV/c]",
            "__energy__":"",
            "__dipole_polarity1__":"",
    }
    beta_limit = 4e3
    el_limit = 25

def main():
    run_dir = "output/rectilinear_cooling_v26/"
    plot_dir = "emittance_plots/"
    run_dir_glob = run_dir+"*0.2*_3*/"
    file_name = "track_beam_amplitude/cooling_test/output.txt"
    cell_length = 2000.0 # full cell length
    file_format = "icool_for009"
    plotter = PlotG4BL(run_dir_glob, file_name, file_format, plot_dir)
    plotter.analysis_list = None
    plotter.e_min = 200.0
    plotter.e_max = 270.0
    plotter.z_range = [55000, 105100]
    plotter.max_station = 2
    plotter.variables_of_interest = ["__dipole_field__", "__momentum__", "__wedge_opening_angle__"]
    plotter.do_plots()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
