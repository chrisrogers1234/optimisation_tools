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
    def __init__(self, run_dir_glob, co_file, reference_file, reference_file_format, plot_dir, max_score):
        self.plot_dir = plot_dir
        self.co_data = []
        self.min_n_cells = 500
        self.target_p = 200
        self.min_p = 169
        self.max_p = 231
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
                if bunch_list[0][0]["p"] > self.max_p or bunch_list[0][0]["p"] < self.min_p:
                    print("Skipping", file_name, f"{self.min_p} > p > {self.max_p}")
                    continue
                new_co_data.append({"bunch_list":bunch_list, "subs":subs_json, "file_name":file_name})
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
        self.figure5 = matplotlib.pyplot.figure()
        self.figure6 = matplotlib.pyplot.figure()
        self.process_data()
        p_list = [abs(data["p_list"][0]-self.target_p) for data in self.co_data]
        closest = p_list.index(min(p_list))
        self.plot_reference_z()
        self.plot_fft(self.co_data[closest])
        self.plot_fit_scan(18, 23)
        self.plot_survival()
        self.plot_phase_space()
        self.figure2.savefig(self.plot_dir+"/fft_max.png")
        self.figure3.savefig(self.plot_dir+"/n_turns.png")
        self.figure4.savefig(self.plot_dir+"/fit_params.png")
        self.figure5.savefig(self.plot_dir+"/survival.png")
        self.figure6.savefig(self.plot_dir+"/phase_space.png")

    def get_ellipse(self, bunch):
        points = [(hit["x"], hit["px"]) for hit in bunch]
        mean, cov = xboa.common.fit_ellipse(points, 1e10)
        print("Fitted")
        print(mean)
        print(cov)
        return cov

    def get_sine(self, x_array, x_mean, tune):
        return [0.0+math.sin(2*math.pi*tune*x) for x in x_array]


    def sine_fit(self, x_array):
        x_mean = numpy.mean(x_array)
        x_array -= x_mean
        n_crossings = 0
        for i, x1 in enumerate(x_array[1:]):
            x0 = x_array[i]
            if x0 > 0 and x1 < 0:
                n_crossings += 1
        n_data = [i for i in range(len(x_array))]
        tune = n_crossings/len(x_array)
        aopt, acov = scipy.optimize.curve_fit(self.get_sine, n_data, x_array, [x_mean, tune], bounds=[[-numpy.inf, 0], [numpy.inf, 1]])
        print("estimated fit", x_mean, tune, "fitted", aopt)
        return tune

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
            data["tune_alt"] = []
            data["n_list"] = []
            data["p_list"] = []
            data["octupole_list"] = []
            data["max_cell"] = []
            data["amp0"] = []
            dz = bunch_list[0][2]["z"] - bunch_list[0][0]["z"]
            data["var"] = self.get_ellipse(bunch_list[4]).tolist()
            print("process data", i, len(bunch_list))
            for bunch in bunch_list[:]:
                x0 = bunch[0]["x"]
                fft_input_data = numpy.array([hit["x"] for hit in bunch[::2]])
                fft_output_data = scipy.fft.fft(fft_input_data)[1:]
                if len(fft_output_data) == 0:
                    continue

                data["max_cell"].append(max([hit["z"] for hit in bunch])/dz)
                data["fft_detail_nu_list"].append([ffti/len(fft_output_data) for ffti in range(len(fft_output_data))])
                data["fft_detail_re_list"].append([numpy.real(x) for x in fft_output_data])
                data["fft_detail_im_list"].append([numpy.imag(x) for x in fft_output_data])
                data["fft_detail_mag_list"].append([numpy.absolute(x) for x in fft_output_data])
                data["fft_detail_angle_list"].append([numpy.angle(x) for x in fft_output_data])


                data["n_list"].append(len(fft_output_data))
                data["p_list"].append(bunch[0]["p"])
                #data["octupole_list"].append(data["subs"]["__octupole_coefficient__"])
                data["x_init_list"].append(bunch[0]["x"])
                fft_max_i = data["fft_detail_mag_list"][-1].index(max(data["fft_detail_mag_list"][-1]))
                data["fft_list"].append(fft_max_i/len(fft_output_data))
                #data["tune_alt"].append(self.sine_fit(fft_input_data)) # doesnt work!!!
                data["fft_angle_list"].append(data["fft_detail_angle_list"][-1][fft_max_i])
                data["amp0"].append(xboa.bunch.Bunch.get_amplitude(bunch, bunch[0], ["x"], data["var"], {"x":0., "px":0.}))

    def plot_fft(self, data):
        matplotlib.pyplot.close(self.figure1)
        self.figure1 = matplotlib.pyplot.figure()
        axes1 = self.figure1.add_subplot(1, 2, 1)
        axes2 = self.figure1.add_subplot(1, 2, 2)
        p0 = round(data["p_list"][0], 1)
        x0_list = [data["x_init_list"][j] for j, nu_list in 
                        enumerate(data["fft_detail_nu_list"]) if data["max_cell"][j] >  self.min_n_cells]
        colors = matplotlib.pyplot.cm.coolwarm
        try:
            norm = matplotlib.colors.Normalize(min(x0_list), max(x0_list))
        except ValueError:
            print(f"No data found for {data['file_name']}")
            return
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
        if not y_data:
            return [0.0, 0.0], None
        aguess = y_data[0], 0
        x_data = [x for x in x_data if x < 15]
        y_data = [y for i, y in enumerate(y_data) if i < len(x_data)]
        aopt, acov = scipy.optimize.curve_fit(self.get_y, x_data, y_data, aguess)
        return aopt, acov

    def get_label(self, data):
        a_label = data["file_name"]
        a_label = a_label.split("/")[2]
        a_label = a_label.split("_")[1:]
        a_label = " ".join(a_label)
        return a_label

    def plot_reference_z(self):
        axes2 = self.figure2.add_subplot(1, 1, 1)
        axes3 = self.figure3.add_subplot(1, 1, 1)
        color_list = [self.color_lambda(data) for data in self.co_data]
        colors = matplotlib.pyplot.cm.coolwarm
        norm = matplotlib.colors.Normalize(min(color_list), max(color_list))
        mappable = matplotlib.cm.ScalarMappable(norm, colors)
        for i, data in enumerate(self.co_data):
            label = self.get_label(data)
            #p_norm = (data["p_list"][0]-min(all_p_list))/(max(all_p_list) - min(all_p_list))
            rgba = mappable.to_rgba(self.color_lambda(data))
            fft_list, fft_angle_list, x_init_list = [], [], []
            for j, n in enumerate(data["max_cell"]):
                x0 = data["x_init_list"][j]
                if n > self.min_n_cells and x0 > 1e-9:
                    fft_list.append(data["fft_list"][j])
                    #fft_list.append(data["tune_alt"][j])
                    fft_angle_list.append(data["fft_angle_list"][j])
                    x_init_list.append(x0)
            if len(fft_list) == 0:
                print(f"No data found for {data['file_name']}")
                data["fit_params"] = [0.0, 0.0]
                continue
            pplot = axes2.plot(fft_list, x_init_list) #, c=rgba)
            axes2.scatter(fft_list[-1], x_init_list[-1], color=pplot[0].get_color())
            aopt, acov = self.fit(x_init_list, fft_list)
            fit_data = [self.get_y(x, aopt[0], aopt[1]) for x in x_init_list]
            axes2.plot(fit_data, x_init_list, c=pplot[0].get_color(), linestyle='--')
            axes3.scatter(data["max_cell"], data["x_init_list"], label=self.get_label(data))
            data["fit_params"] = aopt
            data["fit_cov"] = acov
        self.figure2.colorbar(mappable=mappable, ax=axes2) 
        ylim = axes2.get_ylim()
        for i in range(2, 8):
            axes2.plot([1/i, 1/i], ylim, c='lightgray', linestyle='--')
        axes2.set_ylim(ylim)
        axes2.set_xlim(0.0, 1.0)
        axes2.set_xlabel("$\\nu$")
        axes2.set_ylabel("x$_0$ [mm]")
        axes3.set_xlabel("number of cells")
        axes3.set_ylabel("x$_0$ [mm]")
        axes3.legend()

    def plot_survival(self):
        #axes2 = self.figure5.add_subplot(1, 2, 2)
        axes1 = self.figure5.add_subplot(1, 1, 1)
        x_list = [self.color_lambda(data) for data in self.co_data]
        y_list_1, y_list_2 = [], []
        for data in self.co_data:
            survivors_1 = [x_init for j, x_init in enumerate(data["x_init_list"]) if data["max_cell"][j] > self.min_n_cells]
            if len(survivors_1):
                y_list_1.append(max(survivors_1))
            else:
                y_list_1.append(0)
            survivors_2 = [amp for j, amp in enumerate(data["amp0"]) if data["max_cell"][j] > self.min_n_cells]
            if len(survivors_2):
                y_list_2.append(max(survivors_2))
            else:
                y_list_2.append(0)
        axes1.scatter(x_list, y_list_1)
        axes1.set_ylabel("Maximum surviving x$_0$ [mm]")
        axes1.set_xlim(0, axes1.get_xlim()[1])
        axes1.set_ylim(0, axes1.get_ylim()[1])
        #axes2.scatter(x_list, y_list_2)
        #axes2.set_ylabel("Maximum surviving Amplitude [mm]")
        #axes2.set_xlim(0, axes2.get_xlim()[1])
        #axes2.set_ylim(0, axes2.get_ylim()[1])

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

    def plot_phase_space(self):
        pass

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
    version = "2024-03-28"
    run_dir = "output/demo_v6/"
    run_dir_glob = [run_dir+f"2024-03-28_baseline_pz=200_step=*_tol=*/"]
    plot_dir = run_dir+f"/plot_tune_tol/"

    file_name = "track_beam_amplitude/da_scan/output*.txt"
    co_file_name = "closed_orbits_cache"
    file_format = "icool_for009"
    plotter = PlotG4BL(run_dir_glob, co_file_name, file_name, file_format, plot_dir, 1e9)
    plotter.beta_limit = 1e4
    plotter.max_p = 231
    plotter.min_n_cells = 10
    plotter.color_lambda = lambda data: data["subs"]["__momentum__"]
    plotter.do_plots()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")