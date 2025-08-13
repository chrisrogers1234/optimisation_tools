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
    def __init__(self, run_dir_glob, co_file, cell_length, target_subs, beam_file_name, file_format, plot_dir, max_score):
        self.plot_dir = plot_dir
        self.co_data = []
        self.cell_length = cell_length
        self.target_subs = target_subs # list of subs
        self.pick_list = []
        self.tracking = []
        self.station_to_z_dict = None
        self.file_format = file_format
        self.beam_file_name = beam_file_name
        self.co_file = co_file
        self.run_dir_glob = run_dir_glob
        self.max_score = max_score
        self.will_plot_beta = True
        self.disp_limits = [-200.0, 0]

    def load(self):
        if "root" in self.file_format:
            optimisation_tools.opal_tracking.import_bdsim_definitions()
            import ROOT
        self.load_data(self.run_dir_glob, self.co_file, self.beam_file_name, self.file_format)
        self.parse_substitutions() # check for variables and sort the list
        self.pick_substitutions()
        self.filter_data() # if max_score, reject data with data["errors"] > max_score
        self.colors =  ["C"+str(i) for i in range(len(self.co_data))]
        utilities.clear_dir(self.plot_dir)

    def load_bunch(self, file_name, reference_file_format):
        if reference_file_format in xboa.hit.Hit.file_types():
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

    def load_data(self, run_dir_glob, co_file, reference_file, reference_file_format):
        globble = []
        if type(run_dir_glob) == type([]):
            for run_dir in run_dir_glob:
                globble += glob.glob(run_dir)
        else:
            globble = glob.glob(run_dir_glob)
        if len(globble) == 0:
            raise RuntimeError("Failed to glob from", run_dir_glob)
        for a_dir in sorted(globble):
            new_co_data = []
            a_file_glob = os.path.join(a_dir, reference_file)
            reference_data_glob = sorted(glob.glob(a_file_glob))
            for i, file_name in enumerate(reference_data_glob):
                new_co_data.append({"bunch_list":self.load_bunch(file_name, reference_file_format)})
            co_file_name = os.path.join(a_dir, co_file)
            try:
                fin = open(co_file_name)
                co_data = json.loads(open(co_file_name).read())
                if len(co_data) != len(new_co_data):
                    raise ValueError(f"Found {len(co_data)} transfer maps ({os.path.split(co_file_name)[1]}) and {len(new_co_data)} reference outputs ({os.path.split(reference_file)[1]}) in {co_file_name}")
                for i, co in enumerate(co_data):
                    new_co_data[i].update(co)
                    print("  closed orbit", co["seed"])
            except OSError:
                print("Failed to load", co_file_name)
            self.co_data += new_co_data

    def _sort_lambda(self, data):
        if "variables" not in data:
            return []
        return [data["variables"][key] for key in self.key_list]

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

    def pick_substitutions(self):
        self.pick_list = []
        for i, data in enumerate(self.co_data):
            if "substitutions" not in data:
                continue
            for subs in self.target_subs:
                will_pick = True
                for key, value in subs.items():
                    print(key, value, data["substitutions"][key])
                    if abs(data["substitutions"][key]-value) > 1e-3:
                        will_pick = False
                        break
                if will_pick:
                    self.pick_list.append(i)
                    break

    def filter_data(self):
        if not self.max_score:
            return
        co_data_tmp = []
        for data in self.co_data:
            if "errors" not in data:
                continue
            if sum(data["errors"]) < self.max_score:
                co_data_tmp.append(data)
        self.co_data = co_data_tmp

    def get_label(self, data):
        label = ""
        for var, value in data["variables"].items():
            try:
                if self.key_subs[var] != None:
                    label += self.key_subs[var]+": "+str(value)+" "+self.units_subs[var]
            except KeyError:
                pass
        return label

    def load_beam_file(self, beam_file, file_format):
        bunch_list = xboa.bunch.Bunch.new_list_from_read_builtin(file_format, beam_file)
        print(len(bunch_list))
        self.tracking += bunch_list

    def do_plots(self):
        self.figure1 = matplotlib.pyplot.figure(figsize=(20, 6))
        self.plot_reference_z()
        #self.plot_envelope_z()
        self.figure1.savefig(self.plot_dir+"/reference_z.png")
        self.plot_var2("__momentum__", )
        self.plot_beam_init()
        print("Plotted data with variables", self.key_list)
        max_keys = max([(len(data.keys()), data.keys()) for data in self.co_data])
        print("Top level data keys were", max_keys[1])

    def check_stability(self, data):
        try:
            ref_tm = DecoupledTransferMatrix(self.get_tm(data["tm"]))
        except (ValueError, IndexError):
            return False
        if ref_tm.get_beta(0) > self.beta_limit:
            return False
        if ref_tm.get_beta(1) > self.beta_limit:
            return False
        return True

    def get_tm(self, tracking_matrix):
        if tracking_matrix == None:
            return numpy.zeros((4,4))
        tm = numpy.array([row[1:5] for row in tracking_matrix[0:4]])
        tm = DecoupledTransferMatrix.simplectify(tm)
        return tm

    def get_values(self, held_key_dict, value_lambda):
        """Return list of value_lambda(data) if """
        value_list = []
        for data in self.co_data:
            if "substitutions" not in data:
                continue
            is_equal = True
            for key,value in held_key_dict.items():
                is_equal = is_equal and data["substitutions"][key] == value
            if not is_equal:
                continue
            value_list.append(value_lambda(data))
        return value_list

    def get_key_dict(self):
        key_dict = dict([(key, []) for key in self.key_list])
        for data in self.co_data:
            for key in self.key_list:
                key_dict[key].append(data["substitutions"][key])
        return key_dict

    def plot_var2(self, var_key, held_key = None):
        if held_key:
            group_dict = utilities.get_groups(self.co_data, held_key, "substitutions")
        else:
            group_dict = {"data":{"item_list":[i for i, item in enumerate(self.co_data)]}}
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        for label, item_list in group_dict.items():
            co_data = [self.co_data[i] for i in item_list["item_list"]]
            x_list = []
            beta_0_list = []
            disp_theta_list = []
            disp_0_list = []
            disp_1_list = []
            for i, data in enumerate(co_data):
                if i == 0:
                    data_low = co_data[i]
                    data_high = co_data[i+1]
                elif i == len(co_data)-1:
                    data_low = co_data[i-1]
                    data_high = co_data[i]
                else:
                    data_low = co_data[i-1]
                    data_high = co_data[i+1]
                if self.check_stability(data):
                    ref_tm = DecoupledTransferMatrix(self.get_tm(data["tm"]))
                    beta_0_list.append(ref_tm.get_beta(0))
                    try:
                        z_list, disp_x_list, disp_y_list = self.get_dispersion(data_low, data_high)
                        disp_theta = math.atan2(disp_y_list[0], disp_x_list[0])
                        disp_theta_list.append(math.degrees(disp_theta))
                    except (IndexError, RuntimeError):
                        z_list, disp_x_list, disp_y_list = [None], [None], [None]
                    disp_theta_list.append(None)
                    disp_0_list.append(disp_x_list[0])
                    disp_1_list.append(disp_y_list[0])
                else:
                    print("Failed stability check", i)
                    disp_theta_list.append(None)
                    beta_0_list.append(None)
                    disp_0_list.append(None)
                    disp_1_list.append(None)
                x_list.append(data["substitutions"][var_key])

            if self.will_plot_beta:
                axes.plot(x_list, beta_0_list, label="$\\beta_\\perp$")
                axes.set_xlabel(self.key_subs[var_key]+" "+self.units_subs[var_key])
                axes.set_ylabel("$\\beta$ [mm]")
                axes.set_ylim([0.0, 200.0])
            else:
                axes.plot(x_list, disp_theta_list, label="$\\theta(D)$")
                axes.set_xlabel(self.key_subs[var_key]+" "+self.units_subs[var_key])
                axes.set_ylabel("$\\theta(D)$ [$^\\circ$]")
                axes.set_ylim([-180, 180])
            axes2 = axes.twinx()
            axes2.plot(x_list, disp_0_list, '--', label="D$_x$ ")
            axes2.plot(x_list, disp_1_list, ':', label="D$_y$ ")
            axes2.set_xlabel(self.key_subs[var_key]+" "+self.units_subs[var_key])
            axes2.set_ylabel("Dispersion [mm]")
            axes2.set_ylim(self.disp_limits)
            figure.legend(loc=(0.15, 0.15))
        figure.savefig(os.path.join(self.plot_dir, self.short_form[var_key]+"_vs_beta.png"))

    def plot_var(self):
        key_dict = dict([(key, []) for key in self.key_list])
        beta_0_list, beta_1_list, phi_0_list, phi_1_list = [], [], [], []
        antibeta_0_list, antibeta_1_list = [], []
        x_co_list, y_co_list = [], []
        antix_co_list, antiy_co_list = [], []
        for data in self.co_data:
            antiref_z1 = self.cell_length/4.0
            antiref_z2 = antiref_z1+self.cell_length
            if "substitutions" not in data:
                continue

            for key in self.key_list:
                key_dict[key].append(data["substitutions"][key])
            x_co_list.append(data["ref_track"][0]["x"])
            y_co_list.append(data["ref_track"][0]["y"])
            for zindex1, hit in enumerate(data["ref_track"]):
                if hit["z"] < antiref_z1-0.1:
                    continue
                antix_co_list.append(hit["x"])
                antiy_co_list.append(hit["y"])
                break

            if not self.check_stability(data):
                beta_0_list.append(0)
                beta_1_list.append(0)
                phi_0_list.append(0)
                phi_1_list.append(0)
                antibeta_0_list.append(0)
                antibeta_1_list.append(0)
                continue

            ref_tm = DecoupledTransferMatrix(self.get_tm(data["tm"]))
            beta_0_list.append(ref_tm.get_beta(0))
            beta_1_list.append(ref_tm.get_beta(1))
            phi_0_list.append(ref_tm.get_phase_advance(0))
            phi_1_list.append(ref_tm.get_phase_advance(1))

            for zindex2, hit in enumerate(data["ref_track"]):
                if hit["z"] < antiref_z2-0.1:
                    continue
                break

            tm1 = self.get_tm(data["tm_list"][zindex1])
            tm2 = self.get_tm(data["tm_list"][zindex2])
            anti_tm = numpy.dot(tm2, numpy.linalg.inv(tm1))
            try:
                anti_tm = DecoupledTransferMatrix(anti_tm)
                antibeta_0_list.append(0.0) #anti_tm.get_beta(0))
                antibeta_1_list.append(0.0) #anti_tm.get_beta(1))
            except ValueError:
                antibeta_0_list.append(0.0)
                antibeta_1_list.append(0.0)

        for key in key_dict:
            if self.key_subs[key] == None:
                continue
            x_label = self.key_subs[key]+" "+self.units_subs[key]
            figure = matplotlib.pyplot.figure(figsize=(20, 10))
            x_values = key_dict[key]
            axes = figure.add_subplot(2, 2, 1)
            axes.plot(x_values, x_co_list, c="tab:blue", label="x at z=0", linestyle="dashed")
            axes.plot(x_values, y_co_list, c="tab:blue", label="y at z=0", linestyle="dotted")
            axes.plot(x_values, antix_co_list, c="tab:orange", label="x at z="+str(antiref_z1)+" mm", linestyle="dashed")
            axes.plot(x_values, antiy_co_list, c="tab:orange", label="y at z="+str(antiref_z1)+" mm", linestyle="dotted")
            axes.set_xlabel(x_label)
            axes.set_ylabel("Closed Orbit [mm]")
            axes.legend()

            axes = figure.add_subplot(2, 2, 3)
            axes.plot(x_values, beta_0_list, c="tab:blue", label="$\\beta_0$ at z=0 mm", linestyle="dashed")
            axes.plot(x_values, beta_1_list, c="tab:blue", label="$\\beta_1$ at z=0 mm", linestyle="dotted")
            axes.plot(x_values, antibeta_0_list, c="tab:orange", label="$\\beta_0$ at z="+str(antiref_z1)+" mm", linestyle="dashed")
            axes.plot(x_values, antibeta_1_list, c="tab:orange", label="$\\beta_1$ at z="+str(antiref_z1)+" mm", linestyle="dotted")
            max_beta = round(max(beta_0_list+beta_1_list))
            max_antibeta = round(max(antibeta_0_list+antibeta_1_list))
            beta_str = "Max $\\beta$ "+str(max_beta)+", "+str(max_antibeta)+" mm"
            axes.text(0.1, 0.5, beta_str, transform=axes.transAxes)
            axes.set_xlabel(x_label)
            axes.set_ylabel("$\\beta$ [mm]")
            axes.legend()

            axes = figure.add_subplot(2, 2, 4)
            axes.plot(x_values, phi_0_list, c="b", label="$\\phi_0$", linestyle="dashed")
            axes.plot(x_values, phi_1_list, c="b", label="$\\phi_1$", linestyle="dotted")
            axes.set_xlabel(x_label)
            axes.set_ylabel("Phase Advance [rad]")
            axes.legend()

            print("Phi0: ", phi_0_list, "\nPhi1: ", phi_1_list, "\nBeta0: ", beta_0_list, "\nBeta1: ", beta_1_list)

            figname = key.replace("__", "")
            figname = os.path.join(self.plot_dir, "var_"+figname+".png")
            figure.savefig(figname)

    def plot_beam_init(self):
        print("Found", len(self.tracking), "bunches")
        labels = {"x":"x [mm]", "y":"y [mm]", "px":"p$_x$ [MeV/c]", "py":"p$_y$ [MeV/c]", "p":"p [MeV/c]"}
        for x_var, y_var in ("x", "p"), ("y", "p"), ("px", "p"), ("py", "p") :
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1, 1, 1)
            x_list = [hit[x_var] for hit in self.tracking[0]]
            y_list = [hit[y_var] for hit in self.tracking[0]]
            co_x_list = []
            co_y_list = []
            for data in self.co_data:
                bunch_list = data["bunch_list"]
                print("Found reference bunch list of length", len(bunch_list))
                if not len(bunch_list):
                    continue
                hit = bunch_list[0].get_hits("event_number", 1)[0]
                co_x_list.append(hit[x_var])
                co_y_list.append(hit[y_var])
            #axes.hist2d(x_list+co_x_list, y_list+co_y_list, bins=[100, 100])#, s=1, c="black")
            axes.scatter(co_x_list, co_y_list, c="r")
            axes.set_xlabel(labels[x_var])
            axes.set_ylabel(labels[y_var])
            figure.savefig(os.path.join(self.plot_dir, f"beam_init_{x_var}_vs_{y_var}.png"))

    def plot_reference_z(self):
        axes1 = self.figure1.add_subplot(1, 2, 1)
        axes2 = self.figure1.add_subplot(1, 2, 2)
        z_max = self.cell_length
        plot_data = [data for i, data in enumerate(self.co_data) if i in self.pick_list]
        print("Picked data from", self.pick_list)
        for i, data in enumerate(plot_data):
            bunch_list = data["bunch_list"]
            ref_list = []
            for bunch in bunch_list:
                ref_list += bunch.get_hits("event_number", 1)
            if not len(ref_list):
                print("No data - skipping...")
                continue
            print("Found", len(ref_list), "reference hits")

            c = self.colors[i]
            x_list = [hit["x"] for hit in ref_list if hit["z"] <= z_max]
            y_list = [hit["y"] for hit in ref_list if hit["z"] <= z_max]
            z_list = [hit["z"] for hit in ref_list if hit["z"] <= z_max]
            if i == 0:
                axes1.plot(z_list, x_list, label="x", linestyle='dashed', c="black")
                axes1.plot(z_list, y_list, label="y", linestyle='dotted', c="black")
            label = self.get_label(data)
            axes1.plot(z_list, x_list, label=label, linestyle='dashed', c=c)
            axes1.plot(z_list, y_list, linestyle='dotted', c=c)

            tesla = xboa.common.units["T"]
            b_list = {
                "bx":[hit["bx"]*10/tesla for hit in ref_list if hit["z"] <= z_max],
                "by":[hit["by"]*10/tesla for hit in ref_list if hit["z"] <= z_max],
                "bz":[hit["bz"]/tesla for hit in ref_list if hit["z"] <= z_max],
            }
            plot_list = [("bx", "dashed", "B$_{x}$ [0.1 T]"), ("by", "dotted", "B$_{y}$ [0.1 T]"), ("bz", "solid", "B$_{z}$ [T]")]
            if self.ref_bz:
                b_list["bz_ref"] = [self.ref_bz.get_field(hit["z"]) for hit in ref_list if hit["z"] <= z_max]
                plot_list.append(("bz_ref", "solid", "Ref B$_{z}$ [T]"))
            if i == 0:
                for var, linestyle, y_label in plot_list:
                    axes2.plot(z_list, b_list[var], label=y_label, linestyle=linestyle, c="black")
            for var, linestyle in [("bx", "dashed"), ("by", "dotted"), ("bz", "solid")]:
                if var == "bz":
                    axes2.plot(z_list, b_list[var], label=label, linestyle=linestyle, c=c)
                    max_bz = max([hit["bz"] for hit in ref_list])
                    mean_bz2 = numpy.mean([hit["bz"]**2 for hit in ref_list if hit["z"] <= self.cell_length])
                    axes2.set_title("Max B$_z$: "+str(round(max_bz/tesla, 3))+" T")#; Mean B$_{z}^2$ "+str(round(mean_bz2/tesla, 3))+"T$^2$")
                else:
                    axes2.plot(z_list, b_list[var], linestyle=linestyle, c=c)
        axes1.tick_params(labelsize = self.l_size)

        axes1.set_xlabel("z [mm]", fontsize=self.f_size)
        axes1.set_ylabel("Transverse Position [mm]", fontsize=self.f_size)
        axes1.tick_params(labelsize = self.l_size)
        axes1.legend(loc="upper right")

        axes2.set_xlabel("z [mm]", fontsize=self.f_size)
        axes2.set_ylabel("Magnetic Field", fontsize=self.f_size)
        axes2.tick_params(labelsize = self.l_size)
        axes2.legend(loc="upper right")

    def get_beta(self, data):
        z_list = [hit["z"] for hit in data["ref_track"]]
        beta_0_list = []
        beta_1_list = []
        z1_list = []
        for i0 in range(len(tm_list)):
            z0 = z_list[i0]
            z1 = z0+self.cell_length
            dz_list = [abs(z-z1) for z in z_list]
            i1 = dz_list.index(min(dz_list))
            delta_tm = numpy.dot(tm_list[i1], numpy.linalg.inv(tm_list[i0]))
            try:
                delta_tm = DecoupledTransferMatrix(delta_tm, True)
            except (numpy.linalg.LinAlgError, RuntimeError, ZeroDivisionError):
                continue
            z1_list.append(z0)
            beta_0_list.append(delta_tm.get_beta(0))
            beta_1_list.append(delta_tm.get_beta(1))
            if z0 > self.cell_length:
                break
        return z_list, beta_0_list, beta_1_list

    def get_dispersion(self, data_low, data_high):
        if len(data_high) != len(data_low):
            raise RuntimeError("Wrong error")
        p0 = (data_high["ref_track"][0]["pz"]+data_low["ref_track"][0]["pz"])/2
        dp = (data_high["ref_track"][0]["pz"]-data_low["ref_track"][0]["pz"])
        i_list = range(len(data_high))
        dx_list = [data_high["ref_track"][i]["x"]-data_low["ref_track"][i]["x"] for i in i_list]
        dy_list = [data_high["ref_track"][i]["y"]-data_low["ref_track"][i]["y"] for i in i_list]
        z_list = [data_high["ref_track"][i] for i in i_list]
        dispx_list = [dx*p0/dp for dx in dx_list]
        dispy_list = [dy*p0/dp for dy in dy_list]
        print(f"At p0 {p0} dispersion in x is {dispx_list[0]} and in y is {dispy_list[0]}")
        return z_list, dispx_list, dispy_list

    def suptitle(self):
        pass

    def plot_envelope_z(self):
        axes = self.figure1.add_subplot(2, 2, 3)
        for i, data in enumerate(self.co_data):
            try:
                ref_tm = self.get_tm(data["tm"])
            except KeyError:
                continue
            try:
                ref_tm = DecoupledTransferMatrix(ref_tm, True) # True -> simplectify
            except ValueError:
                continue

            # NOTE that decoupling does not mean anything for tm_list - the 
            # matrices are in general not describing transport in a periodic 
            # system
            z_list = [hit["z"] for hit in data["ref_track"]]
            tm_list = [self.get_tm(tracking_matrix) for tracking_matrix in data["tm_list"]]

            ref_v = ref_tm.get_v_m([1, 1])
            #ref_v = ref_tm.v_t
            ref_m = ref_tm.m

            v_z = numpy.dot(ref_m, ref_v)
            v_z = numpy.dot(v_z, numpy.transpose(ref_m))
            #print("Matrix:\n", ref_m, "\nEllipse:\n", ref_v, "\nTransported Ellipse:\n", v_z)

            v_list = [ref_v]
            for tm in tm_list:
                matrix = tm
                v_z = numpy.dot(matrix, ref_v)
                v_z = numpy.dot(v_z, numpy.transpose(matrix))
                v_list.append(v_z)
                #if sum([v_z[i,i] < 0.0 for i in range(4)]):
                #    print(tm, "\n", v_z, numpy.linalg.det(v_z), "\n", DecoupledTransferMatrix.simplecticity(tm), "\n\n")
            z_list = [bunch[0]["z"] for bunch in data["bunch_list"]]
            print ("z", len(z_list), z_list[:10])
            print ("sx", len(v_list), [v[0, 0] for v in v_list[:10]])
            z_list = z_list[:len(v_list)]
            v_list = v_list[:len(z_list)]

            sigma_x_list = [abs(v[0, 0]) for v in v_list]
            sigma_y_list = [abs(v[2, 2]) for v in v_list]
            c = self.colors[i]
            if i == 0:
                axes.plot(z_list, sigma_x_list, label="$\\sigma_x$", linestyle='dashed', c="black")
                axes.plot(z_list, sigma_y_list, label="$\\sigma_y$", linestyle='dotted', c="black")
            label = self.get_label(data)
            axes.plot(z_list, sigma_x_list, label=label, linestyle='dashed', c=c)
            axes.plot(z_list, sigma_y_list, linestyle='dotted', c=c)

        max_beta = min(self.beta_limit, axes.get_ylim()[1])
        axes.set_ylim([0., max_beta])
        axes.set_xlabel("z [mm]")
        axes.set_ylabel("Approx $\\beta$ [mm]")
        axes.legend(loc="upper right")

    def beta_games(self, tm):
        print(tm.m_evector)
        print(tm.m_evalue)
        beta_1x = tm.m_evector[0, 0]**2
        v1 = 0
        v2 = 0

    key_subs = {
            "__coil_radius__":"r0",
            "__dipole_by__":"B$_{y}$",
            "__dp_pos__":"Dipole Position",
            "__n_particles__":None,
            "__solenoid_field__":None,
            "__optimisation_iteration__":None,
            "__optimisation_score__":None,
            "__momentum__":"p$_{tot}$",
            "__energy__":None,
            "__wedge_opening_angle__":"$\\theta_{wedge}$",
            "__n_cells__":None,
            "__dp_pos__":None,
    }
    units_subs = {
            "__coil_radius__":"[mm]",
            "__dipole_by__":"[T]",
            "__dp_pos__":"[fraction of 2 m]",
            "__momentum__":"[GeV/c]",
            "__wedge_opening_angle__":"$^\\circ$",
            "__energy__":"",
    }
    short_form = {
        "__coil_radius__":"r0",
        "__dipole_by__":"by",
        "__dp_pos__":"dp_pos",
        "__momentum__":"ptot",
        "__energy__":None,
    }
    for i in range(1, 13):
        i_str = "_"+str(i)+"__"
        key_subs["__solenoid_inner_radius"+i_str] = None
        key_subs["__solenoid_outer_radius"+i_str] = None
        key_subs["__solenoid_current"+i_str] = None
    beta_limit = 1e4
    l_size = 14
    f_size = 20

class RefBz():
    def __init__(self, b1, b2, length):
        self.b1 = b1
        self.b2 = b2
        self.length = length

    def get_field(self, z):
        return self.b1*math.sin(2.*math.pi*z/self.length)+self.b2*math.sin(4.*math.pi*z/self.length)

def fname(prefix, params, ignore_globs):
    fname = f"{prefix}"
    for key, value in params:
        print(key, value)
        if ignore_globs and ("*" in value or "?" in value):
            continue
        fname += f"{key}={value};_"
    if fname[-2:] == ";_":
        fname = fname[:-2]
    return fname


def main():
    run_dir = "output/demo_apr25_v028/"
    params = [
        ("pz", "*"),
        ("by", "0.2"),
        ("dp_pos", "0.025"),
        ("dp_length", "100"),
        #("polarity", "++++"),
    ]

    run_dir_glob = fname(f"{run_dir}/", params, False)
    plot_dir = fname(f"{run_dir}/plot_optics_", params, True)
    print("plotting into", plot_dir)
    target_pz = [{"__momentum__":mom} for mom in [0.18, 0.2, 0.22]]

    file_name = "tmp/find_closed_orbits/output_1.txt"
    co_file_name = "closed_orbits_cache"
    if "cell_length" in dict(params):
        cell_length = float(dict(params)["cell_length"]) # full cell length
        h0 = 0.0 #float(dict(params)["harmonic_0"])
    else:
        cell_length = 2000
        h0 = 0
    file_format = "icool_for009"
    #file_format = "bdsim_root_file"
    plotter = PlotG4BL(run_dir_glob, co_file_name, cell_length, target_pz, file_name, file_format, plot_dir, 1e9)
    plotter.ref_bz = RefBz(h0, 0.0, cell_length/2)
    plotter.beta_limit = 1e3
    plotter.station_to_z_dict = dict([(i, i*100) for i in range(101)])
    plotter.params = params
    plotter.load()
    plotter.load_beam_file("lattice/ruihu/beam_stage1.beam", "g4beamline_bl_track_file")
    plotter.do_plots()
    print("Wrote output to", plot_dir)


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
