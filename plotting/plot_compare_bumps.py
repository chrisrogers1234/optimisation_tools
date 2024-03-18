"""
Plot a single closed orbit (once tracking has finished)
"""
import json
import sys
import copy
import os
import math
import argparse
import h5py
import glob
import shutil


import matplotlib
import scipy.interpolate
import numpy

import xboa.common

import pyopal.objects.parser
import pyopal.objects.field
import optimisation_tools.plotting.plot as plot
import optimisation_tools.utils.utilities as utilities
from optimisation_tools.utils import decoupled_transfer_matrix

decoupled_transfer_matrix.DecoupledTransferMatrix.det_tolerance = 1

class MultiPlot(object):
    def __init__(self, plot_dir, dir_list, ref_dir = None):
        self.plot_dir = plot_dir
        self.base_dir_list = sorted(dir_list)
        if ref_dir != None:
            self.base_dir_list.insert(0, ref_dir)
        self.allowed_events = ["ID1"]
        self._phi_last = 0.0
        self.r_deviation_list = []
        self.y_deviation_list = []
        self.da_n_hits_list = []
        self.max_divb_list = []
        self.max_curlb_list = []
        self.tune_0_list = []
        self.tune_1_list = []
        self.tune_ratio_list = []
        self.max_y_deviation_list = []
        self.max_r_deviation_list = []
        self.max_trans_deviation_list = []
        self.target_r_list = []
        self.target_theta_list = []
        self.max_au_list = []
        self.max_av_list = []
        self.max_a4d_list = []
        self.min_au_list = []
        self.min_av_list = []
        self.min_a4d_list = []
        self.delta_ax_list = []
        self.tm_entry = 16 # the entry in the list of transfer matrices for amplitude calculation
        self.target_station = 100

        self.amp_bin_width = 0.002
        self.n_amp_bins = 12
        self.a_survival_dict = {"au":[], "av":[], "a4d":[]}

        self.no_graphics = False
        self.trackOrbit_file = "track_beam/forwards/VerticalSectorFFA-trackOrbit.dat"
        self.lattice_file = "track_beam/forwards/VerticalSectorFFA.tmp"
        self.da_probe_file = "track_beam/forwards/RINGPROBE01.h5"
        self.co_dir = None
        self.injection_orbit_dir = None
        self.injection_orbit_traj = None # trajectory of the closed orbit
        self.injection_orbit_amp = None # amplitude [mum] vs time [ns
        self.delta_ax_station = 4
        self.aa_co_list = []

    def load_orbits(self):
        "Routine to plot things over a long tracking cycle"
        self.orbit_list = []
        for base_dir in self.base_dir_list:
            trackfile = os.path.join(base_dir, self.trackOrbit_file)
            try:
                orbit = plot.LoadOrbit(trackfile, self.allowed_events, None, self.phi_limit, verbose=0)
                self.orbit_list.append(orbit)
            except FileNotFoundError:
                sys.excepthook(*sys.exc_info())

    def transpose_list_of_lists(self, list_of_lists):
        #ref_list = [ref_orbit.orbit[var] for var in ["phi", "r", "z"]]
        list_of_lists_trans = [[list_of_lists[j][i] for j in range(len(list_of_lists))] for i in range(len(list_of_lists[0]))]
        return list_of_lists_trans

    def sorted_ref_list(self):
        ref_list = [self.orbit_list[0].orbit[var_y] for var_y in ["phi", "r", "z"]]
        ref_list = self.transpose_list_of_lists(ref_list)
        ref_list = sorted(ref_list)
        ref_list = self.transpose_list_of_lists(ref_list)
        return ref_list

    def parse_file_name(self, file_name):
        r = file_name.split("=")[1].split("_")[0]
        theta = file_name.split("=")[-1].split("/")[0]
        try:
            theta = float(theta)
        except Exception:
            theta = float(theta.split("_test")[0])
        return float(r)/1000.0, float(theta)

    def get_max_deviations(self):
        self.max_y_deviation_list, self.max_r_deviation_list, self.max_trans_deviation_list = [], [], []
        ref_list = self.sorted_ref_list()
        n_points = len(ref_list[0])
        for orbit in self.orbit_list:
            test_list = orbit.interpolate("phi", ["r", "z"], ref_list[0])
            delta_r_list = [abs(test_list[1][i]-ref_list[1][i]) for i in range(n_points)]
            delta_y_list = [abs(test_list[2][i]-ref_list[2][i]) for i in range(n_points)]
            delta_trans_list = [(delta_r_list[i]**2+delta_y_list[i]**2)**0.5 for i in range(n_points)]
            self.max_y_deviation_list.append(max(delta_y_list))
            self.max_r_deviation_list.append(max(delta_r_list))
            self.max_trans_deviation_list.append(max(delta_trans_list))
            index = delta_trans_list.index(self.max_trans_deviation_list[-1])

    def get_targets(self):
        self.energy_list = [orbit.get_kinetic_energy(0) for orbit in self.orbit_list]
        try:
            self.target_r_list = [self.parse_file_name(orbit.file_name)[0] for orbit in self.orbit_list]
            self.target_theta_list = [self.parse_file_name(orbit.file_name)[1] for orbit in self.orbit_list]
        except:
            self.target_r_list = [i for i, o in enumerate(self.orbit_list)]
            self.target_theta_list = [i for i, o in enumerate(self.orbit_list)]
        self.r_deviation_list = [orbit.interpolate("phi", "r", [108])[1][0] for orbit in self.orbit_list]
        self.y_deviation_list = [orbit.interpolate("phi", "z", [108])[1][0] for orbit in self.orbit_list]

    def get_tunes(self):
        self.co_list = [-1]*len(self.base_dir_list)
        self.tune_0_list = [-1]*len(self.base_dir_list)
        self.tune_1_list = [-1]*len(self.base_dir_list)
        self.tune_ratio_list = [1]*len(self.base_dir_list)
        for i, base_dir in enumerate(self.base_dir_list):
            co_file_name = os.path.join(base_dir, "closed_orbits_cache")
            try:
                co = plot.LoadClosedOrbit(co_file_name, tm_entry=self.tm_entry, tm_is_decoupled=True)
                self.co_list.append(co)
            except Exception:
                sys.excepthook(*sys.exc_info())
                continue
            try:
                self.tune_0_list[i] = co.tm.get_phase_advance(0)/math.pi/2.
            except Exception:
                sys.excepthook(*sys.exc_info())
            try:
                self.tune_1_list[i] = co.tm.get_phase_advance(1)/math.pi/2.
            except Exception:
                sys.excepthook(*sys.exc_info())
            self.tune_ratio_list[i] = self.tune_0_list[i]/self.tune_1_list[i]

    def get_maxwell(self):
        lattice_file = os.path.join(self.base_dir_list[0], self.lattice_file)
        ref_field = plot.GetFields(lattice_file)
        self.max_divb_list = []
        self.max_curlb_list = []
        for orbit in self.orbit_list:
            divb_list = []
            curlb_list = []
            for i in range(len(orbit.orbit["phi"])):
                x, y, z = orbit.orbit["x"][i], orbit.orbit["y"][i], orbit.orbit["z"][i]
                t = 0.0
                divb_list.append(ref_field.get_div_b(x, y, z, t))
                curlb = ref_field.get_curl_b(x, y, z, t)
                curlb = (curlb[0]**2+curlb[1]**2+curlb[2]**2)**0.5
                curlb_list.append(curlb)
            self.max_divb_list.append(max(divb_list))
            self.max_curlb_list.append(max(curlb_list))

    def histo_ratio(self, survivor_data, all_data):
        survivor_hist = [0 for i in range(self.n_amp_bins)]
        all_hist = [0 for i in range(self.n_amp_bins)]
        ratio_hist = [0 for i in range(self.n_amp_bins)]
        for x in survivor_data:
            bin_index = int(x/self.amp_bin_width)
            if bin_index < self.n_amp_bins:
                survivor_hist[bin_index] += 1
        for x in all_data:
            bin_index = int(x/self.amp_bin_width)
            if bin_index < self.n_amp_bins:
                all_hist[bin_index] += 1
        for i in range(self.n_amp_bins):
            if all_hist[i] > 0:
                ratio_hist[i] = survivor_hist[i]/all_hist[i]
        #print("RATIO HIST", ratio_hist)
        return ratio_hist


    def get_dax(self, verbose=False):
        self.delta_ax_list = [0]*len(self.base_dir_list)
        self.aa_co_list = [[None, None]]*len(self.base_dir_list)
        dax_station = self.delta_ax_station
        injection_orbit_name = os.path.join(self.injection_orbit_dir, "closed_orbits_cache")
        inj_co = plot.LoadClosedOrbit(injection_orbit_name, tm_entry=self.tm_entry, tm_is_decoupled=True)
        inj_ref = inj_co.ref_track[self.delta_ax_station]
        for i, base_dir in enumerate(self.base_dir_list):
            co_file_name = os.path.join(base_dir, "closed_orbits_cache")
            try:
                co = plot.LoadClosedOrbit(co_file_name, tm_entry=self.tm_entry, tm_is_decoupled=True)
            except:
                continue
            co_ref = co.ref_track[self.delta_ax_station]
            delta_co = [
                co_ref["x"] - inj_ref["x"],
                co_ref["px"]/co_ref["pz"] - inj_ref["px"]/inj_ref["pz"], 0, 0
            ]
            self.aa_co_list[i] = delta_co[0:2]
            aa = co.tm.coupled_to_action_angle(delta_co)
            self.delta_ax_list[i] = aa[1]*inj_ref["p"]/inj_ref["mass"]
            if verbose:
                print(base_dir, delta_co, self.delta_ax_list[i])
        for point in self.injection_orbit_traj:
            aa = inj_co.tm.coupled_to_action_angle(point[:2]+[0,0])
            point[2] = aa[1]*inj_ref["p"]/inj_ref["mass"]

    def get_da(self):
        self.da_n_hits_list = [0]*len(self.base_dir_list)
        self.max_au_list = [0]*len(self.base_dir_list)
        self.max_av_list = [0]*len(self.base_dir_list)
        self.max_a4d_list = [0]*len(self.base_dir_list)
        self.min_au_list = [0]*len(self.base_dir_list)
        self.min_av_list = [0]*len(self.base_dir_list)
        self.min_a4d_list = [0]*len(self.base_dir_list)
        self.a_survival_dict = {"au":[], "av":[], "a4d":[]}
        self.a_survivor_dict = {"au":[], "av":[], "a4d":[]}
        self.a_all_dict = {"au":[], "av":[], "a4d":[]}

        survival_bin = 0.001
        for i, base_dir in enumerate(self.base_dir_list):
            probe_file = os.path.join(base_dir, self.da_probe_file)
            if not os.path.exists(probe_file):
                print("DOESNT EXIST", probe_file)
                continue
            h5 = plot.LoadH5(probe_file)
            co_file_name = os.path.join(base_dir, "closed_orbits_cache")
            try:
                co = plot.LoadClosedOrbit(co_file_name, tm_entry=self.tm_entry, tm_is_decoupled=True)
            except:
                continue
            #print("Closed orbit", co.ref_track[0])
            h5.set_closed_orbit(co)
            hit_data = [item for item in h5.data if item["station"] == self.target_station]
            survival_ids = {item["id"] for item in hit_data}
            init_data = [item for item in h5.data if item["station"] == 0 and item["id"] in survival_ids]
            inot_data = [item for item in h5.data if item["station"] == 0 and item["id"] not in survival_ids]
            #print("INIT\n  ", init_data[0], "\n  ", init_data[1], "\n  ", init_data[10])
            #print("INOT\n  ", inot_data[0], "\n  ", inot_data[1], "\n  ", inot_data[10])

            self.da_n_hits_list[i] = len(hit_data)
            self.max_au_list[i] = max([0]+[item["au"] for item in init_data])
            self.max_av_list[i] = max([0]+[item["av"] for item in init_data])
            self.max_a4d_list[i] = max([0]+[item["a4d"] for item in init_data])
            #print("LEN inot", len(inot_data))
            if len(inot_data) == 0:
                self.min_au_list[i] = self.max_au_list[-1]
                self.min_av_list[i] = self.max_av_list[-1]
                self.min_a4d_list[i] = self.max_a4d_list[-1]
            else:
                self.min_au_list[i] = min([item["au"] for item in inot_data])
                self.min_av_list[i] = min([item["av"] for item in inot_data])
                self.min_a4d_list[i] = min([item["a4d"] for item in inot_data])
            value_dict = {}
            for key in ["au", "av", "a4d"]:
                survivor_data = [item[key] for item in init_data]
                all_data = survivor_data+[item[key] for item in inot_data]
                ratio_hist = self.histo_ratio(survivor_data, all_data)
                self.a_survival_dict[key] += ratio_hist

    def da_plot(self, hits, not_hits):
        x_list = [hit["au"] for hit in hits]
        y_list = [hit["av"] for hit in hits]
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        axes.scatter(x_list, y_list)

    def phi_limit(self, words):
        phi = math.atan2(words[1], words[3])
        if abs((self._phi_last)-phi) > 6:
            #print("Breaking with phi", phi, "phi last", self._phi_last, "words", words)
            return True
        self._phi_last = phi
        return False

    def grok_x_list(self, x_list):
        if x_list == self.target_r_list:
            x_label = "Nominal bump [m]"
            x_name = "nom_bump"
        elif x_list == self.target_theta_list:
            x_label = "Nominal bump angle [degree]"
            x_name = "nom_bump_angle"
        else:
            raise ValueError("Did not recognise x_list in plots")
        return x_label, x_name

    def plot_1d(self, x_list):
        x_label, x_name = self.grok_x_list(x_list)
        for y_list, y_label, y_name, y_range in [
                (self.r_deviation_list, "Actual horizontal bump [m]", "horizontal_bump", [None, None]),
                (self.y_deviation_list, "Actual vertical bump [m]", "vertical_bump", [None, None]),
                (self.da_n_hits_list, f"Number surviving {self.target_station} turns", "survival", [0.0, None]),
                (self.max_au_list, "Maximum initial A$_x$ of surviving particles [mm]", "max_au", [0.0, 0.25]),
                (self.max_av_list, "Maximum initial A$_y$ of surviving particles [mm]", "max_av", [0.0, 0.25]),
                (self.max_a4d_list, "Maximum initial A$_{4d}$ of surviving particles [mm]", "max_a4d", [0.0, 0.25]),
                #(self.min_au_list, "Minimum initial A$_x$ of lost particles [mm]", "min_au", [0.0, 0.25]),
                #(self.min_av_list, "Minimum initial A$_y$ of lost particles [mm]", "min_av", [0.0, 0.25]),
                (self.min_a4d_list, "Minimum initial A$_{4d}$ of lost particles [mm]", "min_a4d", [0.0, 0.10]),
                #(self.max_divb_list, "Max Div(B) [T/m]", "divb", [0.0, None]),
                #(self.max_curlb_list, "Max |Curl(B)| [T/m]", "curlb", [0.0, None]),
                #(self.tune_0_list, "$\\nu_u$", "nu_u", [0.0, 1.0]),
                #(self.tune_1_list, "$\\nu_v$", "nu_v", [0.0, 1.0]),
                #(self.max_y_deviation_list, "Max vertical deviation [m]", "max_vert", [0.0, None]),
                #(self.max_r_deviation_list, "Max horizontal deviation [m]", "max_hor", [0.0, None]),
                #(self.max_trans_deviation_list, "Max deviation [m]", "max_total", [0.0, None]),
            ]:
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1, 1, 1)
            try:
                axes.plot(x_list, y_list)
                axes.scatter(x_list, y_list)
                print("Plotted", x_label, "vs", y_label)
                print(" ", x_list)
                print(" ", y_list)
            except ValueError:
                print("Failed to plot", x_label, "vs", y_label)
                print(" ", x_list)
                print(" ", y_list)
                matplotlib.pyplot.close(figure)
                continue
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            ylim = list(axes.get_ylim())
            if y_range[0] != None:
                ylim[0] = y_range[0]
            if y_range[1] != None:
                ylim[1] = y_range[1]
            axes.set_ylim(ylim)
            figname = os.path.join(self.plot_dir, x_name+"_vs_"+y_name+".png")
            figure.savefig(figname)
            if self.no_graphics:
                matplotlib.pyplot.close(figure)
            print("Made figure", figname)

    def plot_2d_alt(self, min_z, max_z, z_list, z_text):
        x_list = self.target_r_list
        y_list = self.target_theta_list
        #z_list = self.min_a4d_list
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.set_xlabel("radial bump [m]")
        axes.set_ylabel("gradient bump dp$_r$/p$_\\phi$", labelpad=0.0)
        axes.text(1.25, 1.0, z_text, rotation=90, va="top", transform=axes.transAxes)
        #tricont = axes.tricontourf(x_list, y_list, z_list)
        #figure.colorbar(tricont)
        nx_points, min_x, max_x = plot.PlotUtils.hist_range(x_list)
        ny_points, min_y, max_y = plot.PlotUtils.hist_range(y_list)
        hist = axes.hist2d(x_list, y_list, [nx_points, ny_points], 
                           [[min_x, max_x], [min_y, max_y]], False, 
                           z_list, vmin=min_z, vmax=max_z)
        figure.colorbar(hist[3])
        return figure



    def plot_2d(self, x_list):
        x_label, x_name = self.grok_x_list(x_list)
        for a_hist, y_label, y_name in [
                (self.a_survival_dict["au"], "Initial A$_{u}$ [mm]", "au_ratio"),
                (self.a_survival_dict["av"], "Initial A$_{v}$ [mm]", "av_ratio"),
                (self.a_survival_dict["a4d"], "Initial A$_{4d}$ [mm]", "a4d_ratio")
                ]:
            if not len(self.a_survival_dict["au"]):
                print("Failed to make 2D plot", x_name, "vs", y_name)
                continue
            title = "Survival probability"
            for i, x1 in enumerate(x_list[1:]):
                x0 = x_list[i]
                if x0 == x1 and i+2 < len(x_list):
                    x_list[i+1] = (x_list[i+1]+x_list[i+2])/2.0
                    title = title+("\nNote bin with x="+str(x_list[i])+" has been displaced to "+str(x_list[i+1]))
            x_hist = []
            for x in x_list:
                x_hist += [x for i in range(self.n_amp_bins)]
            y_hist = []
            for x in x_list:
                y_hist += [self.amp_bin_width*(i+0.5) for i in range(self.n_amp_bins)]
            x_bins = [(x_list[0]-x_list[1])/2]+\
                     [(x_list[i]+x_list[i+1])/2 for i, x in enumerate(x_list[:-1])]+\
                     [x_list[-1]+(x_list[-1]-x_list[-2])/2.0]
            y_bins = [i*self.amp_bin_width for i in range(self.n_amp_bins+1)]
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1, 1, 1)
            h2d = axes.hist2d(x_hist, y_hist, [x_bins, y_bins], weights=a_hist)
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            axes.set_title(title)
            figure.colorbar(h2d[3], ax=axes)
            figname = os.path.join(self.plot_dir, x_name+"_vs_"+y_name+".png")
            figure.savefig(figname)
            if self.no_graphics:
                matplotlib.pyplot.close(figure)
            print("Made figure", figname)

    def parse_orbits(self):
        self.get_targets()
        #self.get_max_deviations()
        self.get_tunes()
        #self.get_maxwell()
        self.get_da()
        self.get_dax(True)


class CalculateBumpSettings():
    def __init__(self, plot):
        self.injection_orbit_amp = plot.injection_orbit_amp
        self.injection_orbit_traj = plot.injection_orbit_traj
        self.plot_dir = plot.plot_dir
        self.base_dir_list = plot.base_dir_list
        self.aa_co_list = plot.aa_co_list
        self.subs_file = "track_beam/da/subs.json"
        self.ramp_down_time = -1.0
        self.field_values = dict((f"__h_bump_{i}_field__", list()) for i in range(1, 6))

    def calculate_bump_settings(self):
        self.t_list = [value[1] for value in self.injection_orbit_amp]
        self.amp_list = [value[0] for value in self.injection_orbit_amp]
        self.point_list = [self.get_point(amp) for amp in self.amp_list]
        for field_sub_key in self.field_values.keys():
            points = [self.get_point(amp) for amp in self.amp_list]
            values = self.get_value(field_sub_key, points)
            self.field_values[field_sub_key] = values

    def get_sub(self, sub_key, a_dir):
        """Load a single sub"""
        fin = open(os.path.join(a_dir, self.subs_file))
        json_in = json.loads(fin.read())
        sub = json_in[sub_key]
        return sub

    def get_point(self, amplitude):
        """Map from a desired amplitude to a point in x, x'"""
        x_list = [value[0] for value in self.injection_orbit_traj]
        xp_list = [value[1] for value in self.injection_orbit_traj]
        amplitude_list = [value[2] for value in self.injection_orbit_traj]
        x_interpolator = scipy.interpolate.CubicSpline(amplitude_list, x_list)
        xp_interpolator = scipy.interpolate.CubicSpline(amplitude_list, xp_list)
        x = x_interpolator(amplitude)
        xp = xp_interpolator(amplitude)
        return [float(x), float(xp)]

    def get_value(self, sub_key, required_points):
        """Map from a point in x, x' to a bump magnet setting"""
        value_list = [self.get_sub(sub_key, a_dir) for a_dir in self.base_dir_list[1:]]
        point_list = [value[0:2] for value in self.aa_co_list[1:]]
        value_list = [value for i, value in enumerate(value_list) if point_list[i][0]]
        point_list = [point for i, point in enumerate(point_list) if point_list[i][0]]
        interpolator = scipy.interpolate.LinearNDInterpolator(point_list, value_list)
        for i, x in enumerate(point_list):
            print(x, value_list[i])
        x_points = [value[0] for value in required_points]
        y_points = [value[1] for value in required_points]
        interpolated_values = interpolator(x_points, y_points)
        return interpolated_values

    def plot(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(self.t_list, [value[0] for value in self.point_list])
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("dr [mm]")
        figure.savefig(os.path.join(self.plot_dir, "t_vs_x.png"))
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(self.t_list, [value[1] for value in self.point_list])
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("dr'")
        figure.savefig(os.path.join(self.plot_dir, "t_vs_xp.png"))

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        t_max = self.t_list[-1]+self.ramp_down_time
        times = self.t_list+[t_max]
        for field_key in self.field_values.keys():
            field_name = field_key.replace("_", " ")
            field_name = field_name.rstrip(" ").lstrip(" ")
            values = self.field_values[field_key].tolist()+[0.0]
            axes.plot(times, values, label = field_name)
        axes.legend()
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("Absolute field [T]")
        figure.savefig(os.path.join(self.plot_dir, "bump_fields.png"))

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        for field_key in self.field_values.keys():
            field_name = field_key.replace("_", " ")
            field_name = field_name.rstrip(" ").lstrip(" ")
            values = [value-self.field_values[field_key][0] for value in self.field_values[field_key]]
            axes.plot(times[:-1], values, label = field_name)
        axes.legend()
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("Delta field [T]")
        figure.savefig(os.path.join(self.plot_dir, "delta_bump_fields.png"))

def main():
    do_theta = False
    by="0.1"
    dx="-0.0"
    plot_dir = "output/2023-03-01_baseline/find_bump_v17/"
    ref_dir = os.path.join(plot_dir, "bump=-0.0_by=0.0_bumpp=0.0")
    dir_list = glob.glob(os.path.join(plot_dir, f"bump=*_by={by}_bumpp=*"))
    plot = MultiPlot(plot_dir, dir_list, ref_dir)
    plot.co_dir = os.path.join(plot_dir, "bump=-0.0_by=0.0_bumpp=0.0")
    plot.trackOrbit_file = "track_beam/da/fets_ffa-trackOrbit.dat"
    plot.lattice_file = "track_beam/da/FETS_Ring.tmp"
    plot.da_probe_file = "track_beam/da/ring_probe_001.h5"
    plot.injection_orbit_dir = os.path.join(plot_dir, "bump=-30.0_by=0.1_bumpp=-0.05")
    plot.injection_orbit_traj = [[0.1*i, 0.0, -1.0] for i in range(201)] # trajectory of the closed orbit
    plot.injection_orbit_amp = [[0.0001*i, 250.0*i] for i in range(101)] # amplitude [mm] vs time [ns]
    plot.load_orbits()
    plot.parse_orbits()
    plot_dir = plot_dir+f"/plot_compare_bumps_by={by}_alt/"
    utilities.clear_dir(plot_dir)
    plot.plot_dir = plot_dir
    x_list = plot.target_r_list
    plot.plot_1d(x_list)
    for a_dir in plot.base_dir_list:
        print(a_dir)

    calculator = CalculateBumpSettings(plot)
    calculator.ramp_down_time = 1e4
    calculator.calculate_bump_settings()
    calculator.plot()

    fig = plot.plot_2d_alt(0.0, 0.05, plot.delta_ax_list[1:], "Orbit A$_x$ [mm]")
    fig.savefig(os.path.join(plot.plot_dir, "r_vs_rp_vs_ax_co.png"))

    fig = plot.plot_2d_alt(0.0, 0.1, plot.min_a4d_list[1:], "Min A$_{4D}$ of lost particles [mm]")
    fig.savefig(os.path.join(plot.plot_dir, "r_vs_rp_vs_a4d.png"))

    fig = plot.plot_2d_alt(0.3, 0.5, plot.tune_0_list[1:], "Horizontal tune")
    fig.savefig(os.path.join(plot.plot_dir, "r_vs_rp_vs_tune_x.png"))

    fig = plot.plot_2d_alt(0.3, 0.5, plot.tune_1_list[1:], "Vertical tune")
    fig.savefig(os.path.join(plot.plot_dir, "r_vs_rp_vs_tune_y.png"))

    fig = plot.plot_2d_alt(0.99, 1.01, plot.tune_ratio_list[1:], "Tune ratio")
    fig.savefig(os.path.join(plot.plot_dir, "r_vs_rp_vs_tune_ratio.png"))

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("Press <CR> to finish")

