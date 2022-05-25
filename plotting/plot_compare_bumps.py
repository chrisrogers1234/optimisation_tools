"""
Plot a single closed orbit (once tracking has finished)
"""
import sys
import copy
import os
import math
import argparse
import h5py
import glob
import shutil


import matplotlib
import ROOT

import xboa.common

import PyOpal.parser
import PyOpal.field
import utils.utilities
import plotting.plot as plot


class MultiPlot(object):
    def __init__(self, plot_dir, dir_list, ref_dir = None):
        self.plot_dir = plot_dir
        self.base_dir_list = sorted(dir_list, key=self.parse_file_name)
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
        self.max_y_deviation_list = []
        self.max_r_deviation_list = []
        self.max_trans_deviation_list = []
        self.target_r_list = []
        self.da_n_hits_list = []
        self.max_au_list = []
        self.max_av_list = []
        self.max_a4d_list = []
        self.min_au_list = []
        self.min_av_list = []
        self.min_a4d_list = []

        self.amp_bin_width = 0.002
        self.n_amp_bins = 12
        self.a_survival_dict = {"au":[], "av":[], "a4d":[]}

        self.no_graphics = False

    def load_orbits(self):
        "Routine to plot things over a long tracking cycle"
        self.orbit_list = []
        for base_dir in self.base_dir_list:
            trackfile = os.path.join(base_dir, "track_beam/forwards/VerticalSectorFFA-trackOrbit.dat")
            orbit = plot.LoadOrbit(trackfile, self.allowed_events, self.phi_limit)
            self.orbit_list.append(orbit)

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
        r = file_name.split("track_bump_r_")[1].split("_theta_")[0]
        theta = file_name.split("_theta_")[1].split("/track_beam")[0]
        print ( "PARSE", file_name, theta, theta)
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
            print(self.max_y_deviation_list[-1], self.max_r_deviation_list[-1], self.max_trans_deviation_list[-1])
            index = delta_trans_list.index(self.max_trans_deviation_list[-1])
            print(index, "phi:", test_list[0][index], "dr:", delta_r_list[index], "dy:", delta_y_list[index], "dtot:", delta_trans_list[index])
        print(self.max_y_deviation_list)
        print(self.max_r_deviation_list)
        print(self.max_trans_deviation_list)

    def get_targets(self):
        self.target_r_list = [self.parse_file_name(orbit.file_name)[0] for orbit in self.orbit_list]
        self.target_theta_list = [self.parse_file_name(orbit.file_name)[1] for orbit in self.orbit_list]
        print("GET TARGETS", self.target_r_list, self.target_theta_list)
        self.r_deviation_list = [orbit.interpolate("phi", "r", [108])[1][0] for orbit in self.orbit_list]
        self.y_deviation_list = [orbit.interpolate("phi", "z", [108])[1][0] for orbit in self.orbit_list]

    def get_tunes(self):
        self.co_list = []
        for base_dir in self.base_dir_list:
            co_file_name = os.path.join(base_dir, "closed_orbits_cache")
            self.co_list.append(plot.LoadClosedOrbit(co_file_name))
        self.tune_0_list = [co.tm.get_phase_advance(0)/math.pi/2. for co in self.co_list]
        self.tune_1_list = [co.tm.get_phase_advance(1)/math.pi/2. for co in self.co_list]
        print("Ring tunes", self.tune_0_list, self.tune_1_list)

    def get_maxwell(self):
        lattice_file = os.path.join(self.base_dir_list[0], "track_beam/forwards/VerticalSectorFFA.tmp")
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
        print(self.max_divb_list)
        print(self.max_curlb_list)

    def histo_ratio(self, survivor_data, all_data):
        survivor_hist = [0 for i in range(self.n_amp_bins)]
        all_hist = [0 for i in range(self.n_amp_bins)]
        ratio_hist = [0 for i in range(self.n_amp_bins)]
        for x in survivor_data:
            bin_index = int(x/self.amp_bin_width)
            survivor_hist[bin_index] += 1
        for x in all_data:
            bin_index = int(x/self.amp_bin_width)
            all_hist[bin_index] += 1
        for i in range(self.n_amp_bins):
            if all_hist[i] > 0:
                ratio_hist[i] = survivor_hist[i]/all_hist[i]
        print("RATIO HIST", ratio_hist)
        return ratio_hist

    def get_da(self):
        self.da_n_hits_list = []
        self.max_au_list = []
        self.max_av_list = []
        self.max_a4d_list = []
        self.min_au_list = []
        self.min_av_list = []
        self.min_a4d_list = []
        self.a_survival_dict = {"au":[], "av":[], "a4d":[]}
        self.a_survivor_dict = {"au":[], "av":[], "a4d":[]}
        self.a_all_dict = {"au":[], "av":[], "a4d":[]}

        target_station = 25
        survival_bin = 0.001
        for base_dir in self.base_dir_list:
            probe_file = os.path.join(base_dir, "track_beam/forwards/RINGPROBE01.h5")
            h5 = plot.LoadH5(probe_file)
            co_file_name = os.path.join(base_dir, "closed_orbits_cache")
            co = plot.LoadClosedOrbit(co_file_name)
            h5.set_closed_orbit(co)
            hit_data = [item for item in h5.data if item["station"] == target_station]
            survival_ids = {item["id"] for item in hit_data}
            init_data = [item for item in h5.data if item["station"] == 0 and item["id"] in survival_ids]
            inot_data = [item for item in h5.data if item["station"] == 0 and item["id"] not in survival_ids]

            self.da_n_hits_list.append(len(hit_data))
            self.max_au_list.append(max([item["au"] for item in init_data]))
            self.max_av_list.append(max([item["av"] for item in init_data]))
            self.max_a4d_list.append(max([item["a4d"] for item in init_data]))
            if len(inot_data) == 0:
                self.min_au_list.append(self.max_au_list[-1])
                self.min_av_list.append(self.max_av_list[-1])
                self.min_a4d_list.append(self.max_a4d_list[-1])
            else:
                self.min_au_list.append(min([item["au"] for item in inot_data]))
                self.min_av_list.append(min([item["av"] for item in inot_data]))
                self.min_a4d_list.append(min([item["a4d"] for item in inot_data]))
            value_dict = {}
            for key in ["au", "av", "a4d"]:
                survivor_data = [item[key] for item in init_data]
                all_data = survivor_data+[item[key] for item in inot_data]
                ratio_hist = self.histo_ratio(survivor_data, all_data)
                self.a_survival_dict[key] += ratio_hist

        #self.da_plot([item for item in h5.data if item["station"] == 0], [])
        print(self.da_n_hits_list)
        print(self.max_au_list)
        print(self.max_av_list)
        print(self.max_a4d_list)

    def da_plot(self, hits, not_hits):
        x_list = [hit["au"] for hit in hits]
        y_list = [hit["av"] for hit in hits]
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        axes.scatter(x_list, y_list)

    def phi_limit(self, words):
        phi = math.atan2(words[1], words[3])
        if abs((self._phi_last)-phi) > 6:
            print("Breaking with phi", phi, "phi last", self._phi_last, "words", words)
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
        print("DOING", x_label, x_name)
        return x_label, x_name

    def plot_1d(self, x_list):
        x_label, x_name = self.grok_x_list(x_list)
        for y_list, y_label, y_name, y_range in [
                (self.r_deviation_list, "Actual horizontal bump [m]", "horizontal_bump", [None, None]),
                (self.y_deviation_list, "Actual vertical bump [m]", "vertical_bump", [None, None]),
                (self.da_n_hits_list, "Number surviving 25 turns", "survival", [0.0, None]),
                (self.max_au_list, "Maximum initial A$_u$ of surviving particles [mm]", "max_au", [0.0, 0.025]),
                (self.max_av_list, "Maximum initial A$_v$ of surviving particles [mm]", "max_av", [0.0, 0.025]),
                (self.max_a4d_list, "Maximum initial A$_4d$ of surviving particles [mm]", "max_a4d", [0.0, 0.025]),
                (self.min_au_list, "Minimum initial A$_u$ of lost particles [mm]", "min_au", [0.0, 0.025]),
                (self.min_av_list, "Minimum initial A$_v$ of lost particles [mm]", "min_av", [0.0, 0.025]),
                (self.min_a4d_list, "Minimum initial A$_4d$ of lost particles [mm]", "min_a4d", [0.0, 0.025]),
                (self.max_divb_list, "Max Div(B) [T/m]", "divb", [0.0, None]),
                (self.max_curlb_list, "Max |Curl(B)| [T/m]", "curlb", [0.0, None]),
                (self.tune_0_list, "$\\nu_u$", "nu_u", [None, 0.0]),
                (self.tune_1_list, "$\\nu_v$", "nu_v", [0.0, None]),
                (self.max_y_deviation_list, "Max vertical deviation [m]", "max_vert", [0.0, None]),
                (self.max_r_deviation_list, "Max horizontal deviation [m]", "max_hor", [0.0, None]),
                (self.max_trans_deviation_list, "Max deviation [m]", "max_total", [0.0, None]),
            ]:
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1, 1, 1)
            try:
                axes.plot(x_list, y_list)
                axes.scatter(x_list, y_list)
            except ValueError:
                print("Failed to plot", x_label, "vs", y_label)
                matplotlib.pyplot.close(figure)
                continue
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            ylim = list(axes.get_ylim())
            if y_range[0] != None:
                ylim[0] = y_range[0]
            if y_range[1] != None:
                ylim[1] = y_range[1]
            print(y_name, ylim, y_range)
            axes.set_ylim(ylim)
            figname = os.path.join(self.plot_dir, x_name+"_vs_"+y_name+".png")
            figure.savefig(figname)
            if self.no_graphics:
                matplotlib.pyplot.close(figure)
            print("Made figure", figname)

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
        #self.get_da()

def main():
    do_theta = False
    plot_dir = "output/double_triplet_baseline/single_turn_injection/bump_scan/"
    ref_dir = os.path.join(plot_dir, "track_bump_r_0_theta_0_test3")
    if do_theta:
        dir_list = glob.glob(os.path.join(plot_dir, "track_bump_r_10_theta_*"))
        plot_dir = plot_dir+"/plot_angle_scan/"
    else:
        dir_list = glob.glob(os.path.join(plot_dir, "track_bump_r_*_theta_0"))
        plot_dir = plot_dir+"/plot_position_scan/"
    plot = MultiPlot(plot_dir, dir_list, ref_dir)
    plot.load_orbits()
    plot.parse_orbits()
    if do_theta:
        x_list = plot.target_theta_list
    else:
        x_list = plot.target_r_list
    plot.plot_1d(x_list)
    plot.plot_2d(x_list)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("Press <CR> to finish")

