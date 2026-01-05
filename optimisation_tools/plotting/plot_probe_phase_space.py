import subprocess
import copy
import shutil
import glob
import os
import json
import sys
import operator
import math

import matplotlib
import numpy
import optimisation_tools.opal_tracking

import xboa.common
import xboa.hit

import config.config_base as config
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.twod_transfer_matrix import TwoDTransferMatrix
import optimisation_tools.utils.utilities

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
                #print(hit[-1], test_value, value)
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
        self.accepted = set([hit[-3] for hit in accept_hit_list])
        print("TransmissionCut accepted event numbers: ", self.accepted)

    def will_cut(self, hit, decoupled, aa, ref_hit):
        return hit['event_number'] not in self.accepted

class PlotProbes(object):
    def __init__(self, forwards_file_name_list, probe, plot_dir):
        path = os.path.expandvars("${OPAL_EXE_PATH}/opal")
        mass = xboa.common.pdg_pid_to_mass[2212]
        ref = xboa.hit.Hit.new_from_dict({"pid":2212, "mass":mass, "charge":1.})
        file_name_dict = self.setup_stations(forwards_file_name_list) # setup a dict of file_name:station_number
        print(file_name_dict)
        self.tracking = optimisation_tools.opal_tracking.OpalTracking("", "", ref, file_name_dict, path)
        self.tracking.set_file_format("hdf5")
        self.tracking.name_dict
        self.plot_dir = plot_dir
        my_config = config.Config(-0.0, 0.2, 0.0)
        my_config.tracking["verbose"] = 100
        my_config.tracking["station_dt_tolerance"] = 10.0
        my_config.tracking["dt_tolerance"] = -1.0
        my_config.tracking["analysis_coordinate_system"] = "azimuthal_anticlockwise"
        self.extras = ['t', 'event_number', 'kinetic_energy']
        self.extras_labels = ['time [ns]', 'event number', 'Kinetic Energy [MeV]']
        self.probe_data = optimisation_tools.opal_tracking._opal_tracking.StoreDataInMemory(my_config)
        self.tracking.pass_through_analysis = self.probe_data
        self.tracking.verbose = 0
        self.cut_list_1 = []
        self.cut_list_2 = []
        self.transfer_matrix_type = TwoDTransferMatrix
        self.tm_dict = {}
        self.ref_dict = {}
        self.stations = {}
        self.max_a4d = 0.040
        self.colour_dict = {}
        self.title_text_1 = ""
        self.title_text_2 = ""
        self.fig_list = []
        self.shared_range = []
        self.s = 1
        self.f_size = 15
        self.l_size = 10
        self.ring_tof = 1060
        self.station_by_tof = True
        self.m_index = 1.31
        self.max_amp = []
        self.n_survive = []
        self.var_list = ["x", "x'", "y", "y'"]
        self.do_individual_plots = True
        self.station_lambda = lambda station_list: station_list[0:1000]
        self.plot_space = "physical"
        self.injected_turn = []
        self.verbose_stations = [0, 1]
        self.set_plot_limits(probe)

    def setup_stations(self, file_name_globs):
        file_name_list = []
        for fname in file_name_globs:
            file_name_list += glob.glob(fname)
        endings = [os.path.split(fname)[1] for fname in file_name_list]
        endings = sorted(list(set(endings)))
        station_dict = {}
        ev_counter = {}
        for fname in file_name_list:
            this_ending = os.path.split(fname)[1]
            if this_ending not in ev_counter:
                ev_counter[this_ending] = 0
            ev_counter[this_ending] += 1
            station_dict[fname] = (endings.index(this_ending), ev_counter[this_ending])
        return station_dict


    def load_data(self):
        self.tracking._read_probes()
        for co_params in self.co_param_list:
            self.load_closed_orbits(co_params)
        self.stations = {}
        for hit_list in self.probe_data.last:
            #print("HITLIST LENGTH", len(hit_list))
            for hit in hit_list:
                #print("Add station", hit["station"])
                if self.station_by_tof:
                    hit["station"] = int(hit["t"]/self.ring_tof)
                if hit["station"] not in self.stations:
                    self.stations[hit["station"]] = 0
                self.stations[hit["station"]] += 1
                if hit["event_number"] not in self.colour_dict:
                    self.colour_dict[hit["event_number"]] = hit["station"]
        self.n_events = len(self.colour_dict)
        print("Loaded following stations:number of hits", json.dumps(self.stations, indent=2))
        self.build_colours()

    def will_cut_1(self, hit, decoupled, aa, ref_hit):
        for cut in self.cut_list_1:
            if cut.will_cut(hit, decoupled, aa, ref_hit):
                return True
        return False

    def will_cut_2(self, hit, decoupled, aa, ref_hit):
        for cut in self.cut_list_2:
            if cut.will_cut(hit, decoupled, aa, ref_hit):
                return True
        return False

    def get_ref_tm(self, this_station, hit = None):
        #this_station = hit["station"]
        try:
            raise KeyError("Force station") # in 4-fold lattice we just want to use ring tm, cell tm doesn't make sense
            ref = self.ref_dict[this_station]
            tm = self.tm_dict[this_station]
        except KeyError:
            ref = self.ref_dict[0]
            tm = self.tm_dict[0]
        if hit == None:
            hit = ref
        # dp = p_1.p_0/p_0^2
        dp_over_p = (hit["px"]*ref["px"]+hit["py"]*ref["py"]+hit["pz"]*ref["pz"])/ref["p"]**2-1
        if self.m_index > 1e-9:
            dvert = dp_over_p/self.m_index*1000
            ref["y"] += dvert
        #if hit["station"] < 2:
        #    print("Calculating ref for hit st", hit["station"], "p", hit["p"], "dp_over_p", dp_over_p, "m", self.m_index, "dy", dvert, "y", ref["y"], "hit y", hit["y"])
        return ref, tm


    def get_hits(self, station):
        hit_list_of_lists = self.probe_data.last
        station_hit_list, station_not_list = [], []
        self.cut_list_1.append(Cut("station", station, operator.ne))
        n_cuts = 0
        for i, hit_list in enumerate(hit_list_of_lists):
            for hit in hit_list:
                this_station = int(hit["t"]/self.ring_tof) #hit["station"]
                #print("Testing hit with t", hit["t"], "station", this_station, "vs", station)
                if this_station != station:
                    continue
                ref, tm = self.get_ref_tm(station, hit)
                coupled = [hit[var]-ref[var] for var in self.var_list]
                decoupled = tm.decoupled(coupled).tolist()
                action_angle = tm.coupled_to_action_angle(coupled)
                action_angle[1] *= hit["p"]/hit["mass"]
                action_angle[3] *= hit["p"]/hit["mass"]
                coupled = [hit[var] for var in self.var_list]
                if station in self.verbose_stations and i == 0:
                    print("For station", station)
                    print("  TM M:\n", tm.m)
                    print("  Ref psv:", [format(ref[var], "8.6f") for var in self.var_list])
                if station in self.verbose_stations and i < 3:
                    print(f"  Hit {i} psv:", [format(ref[var], "8.6f") for var in self.var_list],
                          "Delta psv:\n", [hit[var]-ref[var] for var in self.var_list])
                if self.will_cut_1(hit, decoupled, action_angle, hit_list[0]):
                    n_cuts += 1                   
                    continue
                time = hit["t"]/self.ring_tof
                time = (time-0.25 - math.floor(time-0.25))*self.ring_tof
                energy = hit["kinetic_energy"]
                if station == 0 and hit["event_number"] < 3:
                    print("Energy at st1, ev", hit["event_number"], hit["kinetic_energy"], "time", hit["t"])
                energy = (hit["p"]-ref["p"])/ref["p"]
                if self.will_cut_2(hit, decoupled, action_angle, hit_list[0]):
                    station_not_list.append([this_station]+coupled+decoupled+action_angle+[time, energy]+[hit[e] for e in self.extras]+[ref["y"]])
                else:
                    station_hit_list.append([this_station]+coupled+decoupled+action_angle+[time, energy]+[hit[e] for e in self.extras]+[ref["y"]])
                self.extras_labels += "y$_{ref}$ [mm]"
        if False:
            print("Station", station, "not plotted:", n_cuts, "not accepted:", len(station_not_list), "accepted:", len(station_hit_list))
            for hit in station_hit_list[0:2]:
                print("coupled", end=" ")
                for x in hit[1:5]:
                    print(format(x, "12.6g"), end=" ")
                print("decoupled", end=" ")
                for x in hit[5:9]:
                    print(format(x, "12.6g"), end=" ")
                print("aa", end=" ")
                for x in hit[9:13]:
                    print(format(x, "12.6g"), end=" ")
                print("extras", end=" ")
                for x in hit[13:]:
                    print(format(x, "12.6g"), end=" ")
                print()
        del self.cut_list_1[-1]
        return station_hit_list, station_not_list

    def load_closed_orbits(self, co_params):
        fin = open(co_params["filename"])
        co = json.loads(fin.readline())[0]

        ref_to_bump_station_mapping = co_params["ref_to_bump_station_mapping"]
        for hit in co["ref_track"]:
            #print("hit ", [hit[var] for var in ["station", "x", "px", "y", "py"]])
            #print("seed", [co["seed_hit"][var] for var in ["station", "x", "px", "y", "py"]])
            try:
                bump_station = ref_to_bump_station_mapping[hit["station"]]
            except KeyError: # if hit station not in the list, we ignore it
                continue
            if bump_station in self.ref_dict:
                continue
            mass = xboa.common.pdg_pid_to_mass[2212]
            hit.update({"pid":2212, "mass":mass, "charge":1.})
            hit = xboa.hit.Hit.new_from_dict(hit)
            self.ref_dict[bump_station] = hit
            self.ref_dict[bump_station+100] = hit
        tm = [row[1:5] for row in co["tm"]]
        tm = self.transfer_matrix_type(tm, True)
        for var in ref_to_bump_station_mapping.values():
            self.tm_dict[var] = tm
        #tm.print_tests()

    def plot_phase_spaces(self):
        station_list = sorted(list(self.stations.keys()))[::]
        station_list = self.station_lambda(station_list)#+station_list[60::10]#+station_list[0:11:1]+station_list[60::10]+station_list[-2:-1:1]
        station_list = set(station_list)
        for station in station_list:
            h_list, n_list = self.get_hits(station)
            ref, tm = self.get_ref_tm(station)
            print("\rPlotting", len(h_list), "hits and", len(n_list), "misses for station", station, "in", self.plot_space, "coordinates", end="")#, max([h[10]+h[12] for h in h_list]))
            if len(self.colour_dict) == 0:
                # marker size should be appropriate to the station with most hits
                max_points = max(self.stations.values())
                self.s = 2*optimisation_tools.utils.utilities.matplot_marker_size(max_points)
            if len(h_list) == 0:
                print("No hits for station", station)
                #continue
            z_axis = None
            xlim, ylim = None, None
            if len(h_list):
                self.max_amp += [h[1] for h in h_list]
            self.n_survive.append(len(h_list))
            if not self.do_individual_plots:
                continue
            for name, hit_list, not_list in [("", h_list, n_list)]:#, ("not-", [], n_list), ("hit-", h_list, [])]:
                figure, axis_list = optimisation_tools.utils.utilities.setup_da_figure_single(False)
                if self.plot_space == "decoupled":
                    self.plot_phase_space(axis_list[0], hit_list, not_list, 5, 7, z_axis, station)
                    self.plot_phase_space(axis_list[1], hit_list, not_list, 5, 6, z_axis, station)
                    self.plot_phase_space(axis_list[2], hit_list, not_list, 7, 8, z_axis, station)
                elif self.plot_space == "physical":
                    self.plot_phase_space(axis_list[0], hit_list, not_list, 1, 3, z_axis, station)
                    self.plot_phase_space(axis_list[1], hit_list, not_list, 1, 2, z_axis, station)
                    self.plot_phase_space(axis_list[2], hit_list, not_list, 3, 4, z_axis, station)
                    ax_index = 3
                elif self.plot_space == "action_angle":
                    self.plot_phase_space(axis_list[0], hit_list, not_list, 10, 12, z_axis, station)
                    self.plot_phase_space(axis_list[1], hit_list, not_list, 9, 10, z_axis, station)
                    self.plot_phase_space(axis_list[2], hit_list, not_list, 11, 12, z_axis, station)
                else:
                    raise RuntimeError(f"Did not recognise plot space {self.plot_space}")
                if name == "":
                    self.title_text_2 = "showing good and bad hits"
                elif name == "not-":
                    self.title_text_2 = "showing only bad hits"
                elif name == "hit-":
                    self.title_text_2 = "showing only good hits"
                else:
                    self.title_text_2 = name
                #self.suptitle(figure, station, hit_list, not_list)
                station_str = str(station).rjust(3, '0')
                suptitle = "Station: "+str(station)+"\nwith "+str(len(hit_list))+"/"\
                                +str(len(hit_list)+len(not_list))+" good hits and "+str(self.n_events)+" total injected"
                if self.do_suptitle:
                    figure.suptitle(suptitle)
                figure.savefig(self.plot_dir+f"/transverse_station-{self.plot_space}_"+name+station_str+".png")
                self.fig_list.append(figure)
                if station != 0:
                    matplotlib.pyplot.close(figure)
                if self.plot_space == "longitudinal":
                    figure = matplotlib.pyplot.figure(figsize=(10, 6))
                    if self.do_suptitle:
                        figure.suptitle(suptitle)
                    axes = figure.add_subplot(1, 1, 1,  position=[0.15, 0.2, 0.3, 0.7])
                    self.plot_phase_space(axes, hit_list, not_list, 13, 14, z_axis, station)
                    axes = figure.add_subplot(1, 1, 1,  position=[0.6, 0.2, 0.3, 0.3])
                    self.plot_phase_space(axes, hit_list, not_list, 1, 14, z_axis, station)
                    axes.plot(axes.get_ylim(), axes.get_ylim())
                    axes = figure.add_subplot(1, 1, 1,  position=[0.6, 0.6, 0.3, 0.3])
                    self.plot_phase_space(axes, hit_list, not_list, 3, 14, z_axis, station)
                    axes.plot(axes.get_ylim(), axes.get_ylim())
                    figure.savefig(self.plot_dir+"/longitudinal_station-"+name+station_str+".png")
                    self.fig_list.append(figure)
                    matplotlib.pyplot.close(figure)


    def set_plot_limits(self, probe):
        self.lim_dict = {
            1:[3500, 3700],
            2:[-0.00, 0.20],
            3:[-100, 100],
            4:[-0.100, 0.100],
            5:[-70, 70],
            6:[-0.01, 0.01],
            7:[-100, 100],
            8:[-0.01, 0.01],
            9:[-math.pi, math.pi],
            10:[-0.0001, 0.21],
            11:[-math.pi, math.pi],
            12:[-0.0001, 0.21],
            13:[-100.0, self.ring_tof+100.0],
            14:[-0.040, 0.040],
        }
        self.lim_dict[18] = copy.deepcopy(self.lim_dict[3])
        probe_lim_dict = {
            "FOILPROBE_1":{
                1:[3900, 4100],
                2:[0.175, 0.25],
                3:[-30, 30]
            }
        }
        if probe in probe_lim_dict:
            self.lim_dict.update(probe_lim_dict[probe])

    def get_lim(self, var):
        if var in self.lim_dict:
            return self.lim_dict[var]
        return None

    def plot_phase_space(self, axes, hit_list, not_list, x_axis, y_axis, z_axis, station):
        x_list = [hit[x_axis] for hit in hit_list]
        y_list = [hit[y_axis] for hit in hit_list]
        if z_axis != None:
            z_list = [hit[z_axis] for hit in hit_list]
        else:
            z_list = [self.colour(hit) for hit in hit_list]
        #print("Z List:", z_list)
        x_not_list = [hit[x_axis] for hit in not_list]
        y_not_list = [hit[y_axis] for hit in not_list]
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
            9:"$\\phi_x$",
            10:"Norm. A$_x$ [mm]",
            11:"$\\phi_y$",
            12:"Norm. A$_y$ [mm]",
            13:"$\\delta$t [ns]",
            14:"dp/p",
            18:"y$_{ref}$ [mm]"
        }
        for i, label in enumerate(self.extras_labels):
            labels[i+15] = label
        if self.var_list[1] == "x'": # geometric
            labels[2] = "x'"
            labels[4] = "y'"
            labels[6] = "u'"
            labels[8] = "v'"
        name = "phase_space_"+str(station).rjust(3, "0")+"_"+str(x_axis)+"_"+str(y_axis)
        if len(not_list):
            scat = axes.scatter(x_not_list, y_not_list, c='limegreen', edgecolors=None, s=self.s)
        if len(hit_list):
            colors = matplotlib.pyplot.cm.coolwarm
            scat = axes.scatter(x_list, y_list, c=z_list, cmap=colors, vmin=0.0, vmax=1.0, edgecolors=None, s=self.s)
        if x_axis-1 < len(self.var_list) and y_axis-1 < len(self.var_list):
            ref, tm = self.get_ref_tm(station)
            ref_x, ref_y = ref[self.var_list[x_axis-1]], ref[self.var_list[y_axis-1]], 
            scat = axes.scatter([ref_x], [ref_y], c="red", s=self.s)        
        axes.set_xlabel(labels[x_axis], fontsize=self.f_size)
        axes.set_ylabel(labels[y_axis], fontsize=self.f_size)
        axes.grid(True)
        #axes.set_title("y0 = "+str(y_list[0]))
        x_lim, y_lim = self.get_lim(x_axis), self.get_lim(y_axis)
        if x_lim:
            axes.set_xlim(x_lim)
        if y_lim:
            axes.set_ylim(y_lim)
        axes.tick_params(labelsize = self.l_size)
        if z_axis != None:
            axes.get_figure().colorbar(scat)

    def colour(self, hit):
        """Return colour_dict[event_number]"""
        return self.colour_dict[hit[16]] # event number

    def build_colours(self):
        """
        colour_dict maps event_number:(some value). We normalise across the 
        whole colour_dict on the first call to plot_phase_space, so that 
        event number maps to a colour constantly through the whole plotting
        """
        print("Build colours", [(k, v) for k, v in self.colour_dict.items()])
        values = [v for v in self.colour_dict.values()]
        min_value, max_value = min(values), max(values)
        delta_value = max_value-min_value
        if delta_value == 0:
            delta_value = 1
        for key in self.colour_dict:
            self.colour_dict[key] = (self.colour_dict[key] - min_value)/delta_value

    def suptitle(self, figure, station, hit_list, not_hit_list):
        title = "Station "+str(station)+" Good hits: "+str(len(hit_list))+"\n"
        t_list = [hit[-2] for hit in hit_list]
        title += " $\\bar{t}$: "+format(numpy.mean(t_list), "6.2f")+" ns"
        title += " $\\sigma(t)$: "+format(numpy.std(t_list), "6.2f")+" ns"
        if self.title_text_1:
            title += "\n"+self.title_text_1+" "+self.title_text_2

        figure.suptitle(title)


    def movie(self):
        here = os.getcwd()
        os.chdir(self.plot_dir)
        #mencoder mf://turn*.png -mf w=800:h=600:fps=5:type=png -ovc lavc -lavcopts vcodec=msmpeg4:mbd=2:trell -oac copy -o injection.avi
        try:
            output = subprocess.check_output(["mencoder",
                                    "mf://transverse_*.png",
                                    "-mf", "w=800:h=600:type=png", #:fps=1
                                    "-ovc", "lavc",
                                    "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                    "-oac", "copy",
                                    "-o", "transverse_injection.avi"])
        except:
            print("Transverse movie failed")
        try:
            output = subprocess.check_output(["mencoder",
                                    "mf://longitudinal_*.png",
                                    "-mf", "w=800:h=600:type=png", #fps=1:
                                    "-ovc", "lavc",
                                    "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                    "-oac", "copy",
                                    "-o", "longitudinal_injection.avi"])
        except:
            print("Longitudinal movie failed")
        os.chdir(here)

def get_rf_frequency(dir_name):
    ring_tof = 1.10591
    freq = dir_name.split("rf_freq=")[1]
    freq = freq.split("_")[0]
    freq = float(freq)
    return freq*ring_tof

def get_dphi(dir_name):
    phi = dir_name.split("rf_dphi=")[1]
    phi = phi.split("/")[0]
    phi = float(phi)
    return phi

def get_dummy(dir_name):
    return 0

def main(cut_station):
    DecoupledTransferMatrix.det_tolerance = 1.0
    amp_dir = "output/2023-03-01_baseline/find_bump_v17/"
    glob_dir = f"{amp_dir}/bump=-50.0_by=0.2_bumpp=-0.09/track_beam/da/"
    ####### max_amp plot setup
    get_delta = get_dummy
    x_label = "(RF frequency)*(ring tof)"
    max_amp_var = 1
    y_label = "Radius [mm]"
    fname = "radius_vs_frequency"
    delta = []
    max_amp = []
    ########
    n_survive = []
    print("Plotting probes for")
    do_individual_plots = True
    for a_dir in sorted(glob.glob(glob_dir)):
        print("   ", a_dir)
    for a_dir in sorted(glob.glob(glob_dir)):
        for probe in ["ring_probe_001"]:
            plot_dir = a_dir+"/plot_probe_phase/"+probe
            if do_individual_plots:
                if os.path.exists(plot_dir):
                    shutil.rmtree(plot_dir)
                os.makedirs(plot_dir)
            for_glob_name = a_dir+"/"+str(probe)+".h5"
            print("Searching in", for_glob_name)
            forwards_file_name_list = glob.glob(for_glob_name)
            plotter = PlotProbes(forwards_file_name_list, probe, plot_dir)
            plotter.station_lambda = lambda station_list: station_list
            plotter.do_suptitle = False 
            plotter.m_index = 0.0
            plotter.do_individual_plots = do_individual_plots
            plotter.co_param_list = [{
                "filename":"output/2023-03-01_baseline/baseline/closed_orbits_cache",
                "ref_to_bump_station_mapping":dict([(i,i) for i in range(1001)]),
            },]
            try:
                plotter.load_data()
            except IOError:
                print("IOError trying", for_glob_name)
                raise
            cut_hits = plotter.get_hits(cut_station)[0]
            plotter.cut_list_1.append(Cut("pz", 0., operator.lt))
            plotter.cut_list_2.append(TransmissionCut(cut_hits))
            plotter.plot_phase_spaces()
            plotter.plot_space = "action_angle"
            plotter.plot_phase_spaces()
            if do_individual_plots:
                plotter.movie()
            h_tune = 0.413336285851698
            v_tune = 0.3879534802467336
            my_delta = get_delta(a_dir)
            delta += [float(my_delta) for a in plotter.max_amp]
            max_amp += [float(a) for a in plotter.max_amp]
            print(f"Wrote images to {plotter.plot_dir}")
    figure = matplotlib.pyplot.figure(figsize=(20,10))
    axes = figure.add_subplot(1, 2, 1)
    axes.scatter(delta, max_amp)
    ylim = axes.get_ylim()
    axes.plot([h_tune, h_tune], ylim, linestyle=':', color='b')
    axes.plot([v_tune, v_tune], ylim, linestyle=':', color='g')
    #axes.plot([1-h_tune, 1-h_tune], ylim, linestyle='--', color='b')
    #axes.plot([1-v_tune, 1-v_tune], ylim, linestyle='--', color='g')
    #axes.plot([2*h_tune, 2*h_tune], ylim, linestyle='-.', color='b')
    #axes.plot([3*h_tune-1, 3*h_tune-1], ylim, linestyle='-.', color='b')
    axes.set_ylim(-0.05, 1.05)
    axes.set_ylim(ylim)
    axes.set_title(f"{y_label} over about 100 turns")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    figure.savefig(f"{amp_dir}/{fname}.png")
    print()

if __name__ == "__main__":
    main(cut_station=100)
    matplotlib.pyplot.show(block=False)
    #input("Done")
