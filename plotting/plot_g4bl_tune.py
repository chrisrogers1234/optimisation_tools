import math
import json
import glob
import os
import sys

import numpy
numpy.set_printoptions(linewidth=200)
import matplotlib
import matplotlib.colors
import scipy

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
        self.min_n_cells = 90
        self.target_p = 200
        self.max_p = 240
        utilities.clear_dir(plot_dir)
        self.load_data(run_dir_glob, co_file, reference_file, reference_file_format)
        self.color_lambda = lambda data: data["bunch_list"][0][0]["p"]

    def load_data(self, run_dir_glob, co_file, reference_file, reference_file_format):
        globble = []
        if type(run_dir_glob) == type([]):
            for run_dir in run_dir_glob:
                globble += glob.glob(run_dir)
        else:
            globble = glob.glob(run_dir_glob)
        print(run_dir_glob, globble)
        for a_dir in sorted(globble):
            reference_data_glob = sorted(glob.glob(os.path.join(a_dir, reference_file)))
            new_co_data = []
            for i, file_name in enumerate(reference_data_glob):
                bunch_list = Bunch.new_list_from_read_builtin(
                        reference_file_format,
                        file_name,
                        sort_variable = "event_number"
                    )
                dirname = os.path.dirname(file_name)
                subs = open(os.path.join(dirname, "subs.json")).read()
                subs_json = json.loads(subs)
                if bunch_list[0][0]["p"] > self.max_p:
                    print("Skipping", file_name, "p > self.max_p")
                    continue
                new_co_data.append({"bunch_list":bunch_list, "subs":subs_json})
            print(len(new_co_data), os.path.join(a_dir, reference_file))
            self.co_data += new_co_data

    def get_label(self, data):
        label = ""
        return label

    def do_plots(self):
        self.figure1 = matplotlib.pyplot.figure()
        self.figure2 = matplotlib.pyplot.figure()
        self.figure3 = matplotlib.pyplot.figure()
        self.figure4 = matplotlib.pyplot.figure()
        self.process_data()
        p_list = [abs(data["p_list"][0]-self.target_p) for data in self.co_data]
        closest = p_list.index(min(p_list))
        self.plot_reference_z()
        self.plot_fft(self.co_data[closest])
        self.plot_fit_scan(180, 230)
        self.figure2.savefig(self.plot_dir+"/fft_max.png")
        self.figure3.savefig(self.plot_dir+"/n_turns.png")
        self.figure4.savefig(self.plot_dir+"/fit_params.png")

    def process_data(self):
        for i, data in enumerate(self.co_data):
            bunch_list = data["bunch_list"]
            data["x_init_list"] = []
            data["fft_list"] = []
            data["fft_angle_list"] = []
            data["fft_detail_list"] = []
            data["fft_detail_nu_list"] = []
            data["fft_detail_re_list"] = []
            data["fft_detail_im_list"] = []
            data["fft_detail_mag_list"] = []
            data["fft_detail_angle_list"] = []
            data["n_list"] = []
            data["p_list"] = []
            data["octupole_list"] = []
            for bunch in bunch_list[:]:
                x0 = bunch[0]["x"]
                fft_input_data = [hit["x"] for hit in bunch]
                fft_output_data = scipy.fft.fft(fft_input_data)[1:]
                if len(fft_output_data) == 0:
                    continue

                data["fft_detail_nu_list"].append([ffti/len(fft_output_data) for ffti in range(len(fft_output_data))])
                data["fft_detail_re_list"].append([numpy.real(x) for x in fft_output_data])
                data["fft_detail_im_list"].append([numpy.imag(x) for x in fft_output_data])
                data["fft_detail_mag_list"].append([numpy.absolute(x) for x in fft_output_data])
                data["fft_detail_angle_list"].append([numpy.angle(x) for x in fft_output_data])

                data["n_list"].append(len(fft_output_data))
                data["p_list"].append(bunch[0]["p"])
                data["octupole_list"].append(data["subs"]["__octupole_coefficient__"])
                data["x_init_list"].append(bunch[0]["x"])
                fft_max_i = data["fft_detail_re_list"][-1].index(max(data["fft_detail_re_list"][-1]))
                data["fft_list"].append(0.5-fft_max_i/len(fft_output_data))
                data["fft_angle_list"].append(data["fft_detail_angle_list"][-1][fft_max_i])

    def plot_fft(self, data):
        matplotlib.pyplot.close(self.figure1)
        self.figure1 = matplotlib.pyplot.figure()
        axes1 = self.figure1.add_subplot(1, 2, 1)
        axes2 = self.figure1.add_subplot(1, 2, 2)
        #axes3 = self.figure1.add_subplot(2, 2, 3)
        p0 = round(data["p_list"][0], 1)
        x0_list = [data["x_init_list"][j] for j, nu_list in 
                        enumerate(data["fft_detail_nu_list"]) if len(nu_list) >  self.min_n_cells]

        colors = matplotlib.pyplot.cm.coolwarm
        norm = matplotlib.colors.Normalize(min(x0_list), max(x0_list))
        mappable = matplotlib.cm.ScalarMappable(norm, colors)
        for j, nu in enumerate(data["fft_detail_nu_list"]):
            if len(data["fft_detail_nu_list"][j]) < self.min_n_cells or \
               (data["x_init_list"][j] < 40 and data["x_init_list"][j] > 5) or \
               data["x_init_list"][j] < 1e-9 or data["x_init_list"][j] > 45:
                continue
            x0 = data["x_init_list"][j]
            rgba = mappable.to_rgba(x0)
            x_axis = data["fft_detail_nu_list"][j]
            y_axis = data["fft_detail_re_list"][j]
            y_axis = [y/max(y_axis) for y in y_axis]
            axes1.plot(x_axis, y_axis, c=rgba, label = f"x$_0$ = {x0} mm")
            y_axis = data["fft_detail_im_list"][j]
            y_axis = [y/max(y_axis) for y in y_axis]
            axes2.plot(x_axis, y_axis, c=rgba, label = f"x$_0$ = {x0} mm")
        axes1.set_xlabel("$\\nu$")
        axes1.set_ylabel("Re(fft($\\nu$))")
        axes1.set_xlim(0, 2.0)
        axes1.legend()
        axes2.set_xlabel("$\\nu$")
        axes2.set_ylabel("Im(fft($\\nu$))")
        axes2.set_xlim(0, 2.0)
        axes2.legend()
        self.figure1.suptitle(f"FFT for p$_{0}$={p0} MeV/c")
        self.figure1.savefig(self.plot_dir+"/fft_z_"+str(p0)+".png")

    def get_y(self, x, a0, a2):
        y = a0+a2*x**2 # action ~ x**0.5
        return y

    def fit(self, x_data, y_data):
        aguess = y_data[0], 0
        x_data = [x for x in x_data if x < 30]
        y_data = [y for i, y in enumerate(y_data) if i < len(x_data)]
        aopt, acov = scipy.optimize.curve_fit(self.get_y, x_data, y_data, aguess)
        return aopt, acov

    def plot_reference_z(self):
        axes2 = self.figure2.add_subplot(1, 1, 1)
        axes3 = self.figure3.add_subplot(1, 1, 1)
        all_p_list = [self.color_lambda(data) for data in self.co_data]
        colors = matplotlib.pyplot.cm.coolwarm
        norm = matplotlib.colors.Normalize(min(all_p_list), max(all_p_list))
        mappable = matplotlib.cm.ScalarMappable(norm, colors)
        for i, data in enumerate(self.co_data):
            label = self.get_label(data)
            #p_norm = (data["p_list"][0]-min(all_p_list))/(max(all_p_list) - min(all_p_list))
            rgba = mappable.to_rgba(self.color_lambda(data))
            fft_list, fft_angle_list, x_init_list = [], [], []
            for j, n in enumerate(data["n_list"]):
                x0 = data["x_init_list"][j]
                if n > self.min_n_cells and x0 > 1e-9:
                    fft_list.append(data["fft_list"][j])
                    fft_angle_list.append(data["fft_angle_list"][j])
                    x_init_list.append(x0)
            pplot = axes2.plot(fft_list, x_init_list, c=rgba)
            aopt, acov = self.fit(x_init_list, fft_list)
            fit_data = [self.get_y(x, aopt[0], aopt[1]) for x in x_init_list]
            axes2.plot(fit_data, x_init_list, c=rgba, linestyle='--')
            axes3.scatter(data["n_list"], data["x_init_list"])
            data["fit_params"] = aopt
            data["fit_cov"] = acov
        self.figure2.colorbar(mappable=mappable, ax=axes2) 
        ylim = axes2.get_ylim()
        for i in range(2, 8):
            axes2.plot([1/i, 1/i], ylim, c='lightgray', linestyle='--')
        axes2.set_ylim(ylim)
        axes2.set_xlim(0.0, 0.5)
        axes2.set_xlabel("$\\nu$")
        axes2.set_ylabel("x$_0$ [mm]")

    def plot_fit_scan(self, pmin, pmax):
        axes = self.figure4.add_subplot(1, 1, 1)
        a0_list = [data["fit_params"][0] for data in self.co_data]
        a1_list = [data["fit_params"][1]*1e4 for data in self.co_data]
        p_list = [data["p_list"][0] for data in self.co_data]
        print("a0:", a0_list)
        print("a1:", a1_list)
        axes.plot(p_list, a0_list, label="$a_0$")
        axes.plot(p_list, a1_list, label="$a_2 \\times 1e4$")
        axes.set_xlabel("P [MeV/c]")
        axes.set_ylabel("Coefficient")
        axes.set_xlim(pmin, pmax)
        axes.set_ylim(0, axes.get_ylim()[1])
        axes.legend()

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

def main():
    run_dir = "output/rectilinear_cooling_v6/"
    run_dir_glob = [run_dir+"oct=*_pz=*/"]#, run_dir+"by=0.2_pz=200_r0=*/", run_dir+"by=0.2_pz=220_r0=*/"]
    plot_dir = run_dir+"/plot_momentum/"

    #run_dir_glob = [run_dir+"by=0.05_pz=?00/", run_dir+"by=0.05_pz=60/"]
    #plot_dir = run_dir+"/scan_plots_momentum_restricted/"
    file_name = "track_beam_amplitude/da_scan/output*.txt"
    co_file_name = "closed_orbits_cache"
    cell_length = 2000.0 # full cell length
    file_format = "icool_for009"
    plotter = PlotG4BL(run_dir_glob, co_file_name, cell_length, file_name, file_format, plot_dir, 1e9)
    plotter.beta_limit = 1e4
    plotter.color_lambda = lambda data: data["subs"]["__octupole_coefficient__"]
    plotter.color_lambda = lambda data: data["subs"]["__momentum__"]
    plotter.do_plots()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")