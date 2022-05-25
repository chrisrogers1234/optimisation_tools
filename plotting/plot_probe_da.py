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
import optimisation_tools.utils.utilities

class PlotProbes(object):
    def __init__(self, forwards_file_name_list, plot_dir):
        path = os.path.expandvars("${OPAL_EXE_PATH}/opal")
        mass = xboa.common.pdg_pid_to_mass[2212]
        ref = xboa.hit.Hit.new_from_dict({"pid":2212, "mass":mass, "charge":1.})
        file_name_dict = self.setup_stations(forwards_file_name_list) # setup a dict of file_name:station_number
        print(file_name_dict)
        self.tracking = optimisation_tools.opal_tracking.OpalTracking("", "", ref, file_name_dict, path)
        self.tracking.set_file_format("hdf5")
        self.tracking.name_dict
        self.plot_dir = plot_dir
        my_config = config.Config()
        my_config.tracking["verbose"] = 100
        my_config.tracking["station_dt_tolerance"] = 1000.0
        my_config.tracking["dt_tolerance"] = -1.0
        my_config.tracking["analysis_coordinate_system"] = "azimuthal"
        self.probe_data = optimisation_tools.opal_tracking._opal_tracking.StoreDataInMemory(my_config)
        self.tracking.pass_through_analysis = self.probe_data
        self.tm_dict = {}
        self.ref_dict = {}
        self.stations = {}
        self.max_a4d = 0.040
        self.color_dict = {}
        self.title_text_1 = ""
        self.title_text_2 = ""
        self.fig_list = []
        self.shared_range = []
        self.verbose = 0
        self.s = 1
        self.f_size = 20
        self.l_size = 14

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
            for hit in hit_list:
                if hit["station"] not in self.stations:
                    self.stations[hit["station"]] = 0
                self.stations[hit["station"]] += 1
        if self.verbose:
            print("Loaded following stations:number of hits", json.dumps(self.stations, indent=2))

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

    def get_hits(self, station):
        hit_list_of_lists = self.probe_data.last
        n_cuts = 0
        extras = ['t', 'event_number']
        station_hit_list = []
        ev_number = 0
        for hit_list in hit_list_of_lists:
            for hit in hit_list:
                if abs(hit["x"] - 4000.0) > 500.0 or abs(hit["y"]) > 500:
                    continue
                this_station = hit["station"]
                try:
                    ref = self.ref_dict[this_station]
                    tm = self.tm_dict[this_station]
                except KeyError:
                    ref = self.ref_dict[0]
                    tm = self.tm_dict[0]
                coupled = [hit[var]-ref[var] for var in ["x", "x'", "y", "y'"]]
                decoupled = tm.decoupled(coupled).tolist()
                action_angle = tm.coupled_to_action_angle(coupled)
                action_angle[1] *= hit["p"]/hit["mass"]
                action_angle[3] *= hit["p"]/hit["mass"]
                coupled = [this_station]+[hit[var] for var in ["x", "x'", "y", "y'"]]
                station_hit_list.append(coupled+decoupled+action_angle+[0, ev_number])
                ev_number += 1
        if self.verbose:
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
        return station_hit_list

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
        tm = DecoupledTransferMatrix(tm, True)
        print("Coupled  \n", tm.m)
        print("Rotation \n", tm.r)
        print("Decoupled\n", tm.t)
        for var in ref_to_bump_station_mapping.values():
            self.tm_dict[var] = tm
        print("TUNE", tm.get_phase_advance(0)/math.pi, tm.get_phase_advance(1)/math.pi)

    def list_min_max(self, hit_list, var_in, var_test):
        var_range = 0
        for var in var_test:
            var_list = [hit[var] for hit in hit_list]
            var_min = min(var_list)
            var_max = max(var_list)
            var_range = max(var_max-var_min, var_range)
        mean = numpy.mean([hit[var_in] for hit in hit_list])
        min_max = [mean-var_range*1.1/2.0, mean+var_range*1.1/2.0]
        return min_max

    def plot_phase_spaces(self):
        station_list = sorted(list(self.stations.keys()))
        hit_list, not_list = [], []
        for station in station_list:
            my_h_list = self.get_hits(station)
            hit_list += my_h_list
        if len(self.color_dict) == 0:
            self.build_colors(hit_list)
        if len(my_h_list) == 0:
            print("No hits for station", station)
            return
        #self.plot_4d_amplitude(hit_list)
        self.s = optimisation_tools.utils.utilities.matplot_marker_size(hit_list)
        z_axis = None
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        max_amplitude = max([hit[10] for hit in hit_list]+[hit[12] for hit in hit_list])
        self.axis_range = [
            None,
            self.list_min_max(hit_list, 1, [1, 3]),
            self.list_min_max(hit_list, 2, [2, 4]),
            self.list_min_max(hit_list, 3, [1, 3]),
            [-0.04, 0.04], #self.list_min_max(hit_list, 4, [2, 4]),
            self.list_min_max(hit_list, 5, [5, 7]),
            [-0.04, 0.04], #self.list_min_max(hit_list, 6, [6, 8]),
            self.list_min_max(hit_list, 7, [5, 7]),
            self.list_min_max(hit_list, 8, [6, 8]),
            None,
            [0.0, max_amplitude*1.1],
            None,
            [0.0, max_amplitude*1.1],
        ]

        figure, axis_list = optimisation_tools.utils.utilities.setup_da_figure(False)

        self.plot_phase_space(axis_list[0], hit_list, not_list, 1, 3, z_axis, station, False)
        self.plot_phase_space(axis_list[1], hit_list, not_list, 1, 2, z_axis, station, True)
        self.plot_phase_space(axis_list[2], hit_list, not_list, 3, 4, z_axis, station, True)
        self.plot_phase_space(axis_list[3], hit_list, not_list, 5, 7, z_axis, station, False)
        self.plot_phase_space(axis_list[4], hit_list, not_list, 5, 6, z_axis, station, True)
        self.plot_phase_space(axis_list[5], hit_list, not_list, 7, 8, z_axis, station, True)
        self.plot_phase_space(axis_list[6], hit_list, not_list, 10, 12, z_axis, station, False)
        self.plot_phase_space(axis_list[7], hit_list, not_list, 9, 10, z_axis, station, False)
        self.plot_phase_space(axis_list[8], hit_list, not_list, 11, 12, z_axis, station, False)
        #self.suptitle(figure, station, hit_list, not_list)
        figure.savefig(self.plot_dir+"/phase-space_station"+str(station)+".png")
        self.fig_list.append(figure)


    def plot_phase_space(self, axes, hit_list, not_list, x_axis, y_axis, z_axis, station, do_ellipse, xlim = None, ylim = None):
        x_list = [hit[x_axis] for hit in hit_list]
        y_list = [hit[y_axis] for hit in hit_list]
        if z_axis != None:
            z_list = [hit[z_axis] for hit in hit_list]
        else:
            z_list = [self.color(hit) for hit in hit_list]

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
            2:"x'",
            3:"y [mm]",
            4:"y'",
            5:"u",
            6:"p$_u$",
            7:"v",
            8:"p$_v$",
            9:"$\\phi_u$",
            10:"Norm. A$_u$ [mm]",
            11:"$\\phi_v$",
            12:"Norm. A$_v$ [mm]",
        }
        if True:
            labels[2] = "x'"
            labels[4] = "y'"
            labels[6] = "u'"
            labels[8] = "v'"
        name = "phase_space_"+str(station).rjust(3, "0")+"_"+str(x_axis)+"_"+str(y_axis)
        if len(not_list):
            scat = axes.scatter(x_not_list, y_not_list, c='white', edgecolors='black', s=self.s)
        if len(hit_list):
            scat = axes.scatter(x_list, y_list, c=z_list, edgecolors=None, s=self.s)
        if do_ellipse:
            if self.verbose:
                print("Fit ellipse for ", labels[x_axis], "vs", labels[y_axis])
            cov, mean = self.fit_ellipse(axes, [x_list, y_list])
            if self.verbose:
                print("  Fitted ellipse centre:\n", mean, "\nCov:\n", cov, "\n")
            if cov is not None:
                #self.draw_ellipse(axes, cov, mean, 'green')
                if x_axis == 1:
                    cov2 = copy.deepcopy(self.tm_dict[hit_list[0][0]].get_v_m([1,1])[0:2, 0:2])
                elif x_axis == 3:
                    cov2 = copy.deepcopy(self.tm_dict[hit_list[0][0]].get_v_m([1,1])[2:4, 2:4])
                elif x_axis == 5:
                    cov2 = copy.deepcopy(self.tm_dict[hit_list[0][0]].v_t[0:2, 0:2])
                elif x_axis == 7:
                    cov2 = copy.deepcopy(self.tm_dict[hit_list[0][0]].v_t[2:4, 2:4])
                cov2 = cov2*(numpy.linalg.det(cov)/numpy.linalg.det(cov2))**0.5
                self.draw_ellipse(axes, cov2, [0,0], 'xkcd:light blue')
        if xlim != None:
            axes.set_xlim(xlim)
        elif self.axis_range[x_axis] != None:
            axes.set_xlim(self.axis_range[x_axis])
        if ylim != None:
            axes.set_ylim(ylim)
        elif self.axis_range[y_axis] != None:
            axes.set_ylim(self.axis_range[y_axis])
        axes.set_xlabel(labels[x_axis], fontsize=self.f_size)
        axes.set_ylabel(labels[y_axis], fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        if z_axis != None:
            axes.get_figure().colorbar(scat)

    def plot_4d_amplitude(self, hit_list):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        decoupled_cov_inv = numpy.linalg.inv(self.tm_dict[hit_list[0][0]].v_t)
        decoupled_mean = [0, 0, 0, 0]
        coupled_cov_inv = numpy.linalg.inv(self.tm_dict[hit_list[0][0]].get_v_m([1,1]))
        coupled_mean = [0, 0, 0, 0]
        coupled = numpy.array([hit[1:5] for hit in hit_list])
        mean = [numpy.mean(coupled.transpose()[i]) for i in range(4)]
        #print("Mean", mean)
        for hit in hit_list:
            decoupled_vec = hit[5:9]
            decoupled_a4d = self.get_amplitude_4d(decoupled_vec, decoupled_cov_inv, decoupled_mean)
            coupled_vec = hit[1:5]
            coupled_a4d = self.get_amplitude_4d(coupled_vec, coupled_cov_inv, coupled_mean)
            #print(decoupled_a4d, coupled_a4d)

    def color(self, hit):
        return self.color_dict[hit[14]] # event number

    def color_hit(self, hit):
        au = hit[10]
        av = hit[12]
        hue = math.atan2(au, av)/(math.pi/2)*(1./3.)+2./3. # range between 0 and 2/3 i.e. green through red through blue
        saturation = min(1, 0.2+0.8*(au+av)/self.max_a4d)
        value = 0.9
        rgb = matplotlib.colors.hsv_to_rgb([hue, saturation, value]) 
        return rgb

    def build_colors(self, hit_list):
        self.color_dict = {}
        a4d_list = [hit[10]+hit[12] for hit in hit_list]
        self.max_a4d = max(a4d_list)
        for hit in hit_list:
            self.color_dict[hit[14]] = self.color_hit(hit)

    def suptitle(self, figure, station, hit_list, not_hit_list):
        title = "Station "+str(station)+" Good hits: "+str(len(hit_list))+"\n"
        t_list = [hit[-2] for hit in hit_list]
        title += " $\\bar{t}$: "+format(numpy.mean(t_list), "6.2f")+" ns"
        title += " $\\sigma(t)$: "+format(numpy.std(t_list), "6.2f")+" ns"
        if self.title_text_1:
            title += "\n"+self.title_text_1+" "+self.title_text_2

        figure.suptitle(title)

    def get_amplitude_4d(self, u_vec, cov_inv, mean):
        vec = numpy.array(u_vec)
        vec = vec-mean
        norm = numpy.dot(cov_inv, numpy.transpose(vec))
        norm = numpy.dot(vec, norm)
        return norm


    def get_amplitude(self, x, y, cov_inv, mean):
        vec = numpy.array([x-mean[0], y-mean[1]])
        norm = numpy.dot(cov_inv, numpy.transpose(vec))
        norm = numpy.dot(vec, norm)
        return norm

    def fit_ellipse(self, axes, points, fraction=0.95):
        n_total = len(points[0])
        points_inside = copy.deepcopy(points)
        while True:
            mean = [numpy.mean(points_inside[0]), numpy.mean(points_inside[1])]
            cov = numpy.cov(points_inside)
            try:
                cov_inv = numpy.linalg.inv(cov)
            except Exception:
                print("Exception while inverting matrix")
                return None, None
            norm_list = []
            for i in range(len(points_inside[0])):
                norm = self.get_amplitude(points_inside[0][i], points_inside[1][i], cov_inv, mean)
                norm_list.append((norm, i))
            norm_list = sorted(norm_list)
            number_left = (fraction*n_total+len(points_inside[0]))/2
            cov *= max(norm_list)[0]
            cov_inv /= max(norm_list)[0]
            if True or self.verbose:
                print("Keep", number_left, "of", len(points_inside[0]))
            if int(number_left) <= len(points_inside[0]):
                break

            for i in range(int(number_to_remove)):
                del_index = norm_list[i][1]
                del points_inside[0][del_index]
                del points_inside[1][del_index]

        #print("Norm list", [norm for (norm, i) in norm_list], "\n\n")
        return cov, mean

    def draw_ellipse(self, axes, cov, mean, color):
        x_list, y_list = [], []

        std = [cov[0, 0]**0.5, cov[1, 1]**0.5]
        try:
            cov_inv = numpy.linalg.inv(cov)
        except Exception:
            print("Failed to plot cov - singular matrix")
            return
        for theta in range(0, 361, 1):
            x = math.sin(math.radians(theta))*std[0]
            y = math.cos(math.radians(theta))*std[1]
            norm = self.get_amplitude(x, y, cov_inv, [0, 0])
            x = x/norm**0.5
            y = y/norm**0.5
            vec = numpy.array([x, y])
            renorm = numpy.dot(cov_inv, numpy.transpose(vec))
            renorm = numpy.dot(vec, renorm)
            x_list.append(x+mean[0])
            y_list.append(y+mean[1])
        #axes.set_title("Ellipse area "+format(numpy.linalg.det(cov)**0.5, "6.4g")+"$\\pi$ mm")
        axes.plot(x_list, y_list, color=color)
        if self.verbose:
            print("Mean", mean)
            print(cov)
            print("Determinant", format(numpy.linalg.det(cov), "6.4g"), "sqrt", format(numpy.linalg.det(cov)**0.5, "6.4g"))
        return cov


def main():
    DecoupledTransferMatrix.det_tolerance = 1.0
    #base_dir = "output/arctan_baseline/baseline_test_2/"
    #for a_dir in glob.glob(os.path.join(base_dir, "find_da/x_9")): #, "track_beam/forwards")):
    #    for probe in ["*"]: #range(1, 3):
    base_dir = "output/2022-03-01_baseline/correlated_painting/tracking_v2"
    globble = os.path.join(base_dir, "track_beam/injected_beam")
    print(globble)
    for a_dir in glob.glob(globble):
        for probe in ["RINGPROBE01", "FOILPROBE_1"]: #range(1, 3):
            plot_dir = a_dir+"/plot_probe_da/"+probe+"/"
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            os.makedirs(plot_dir)
            for_glob_name = a_dir+"/"+probe+".h5"
            forwards_file_name_list = glob.glob(for_glob_name)
            plotter = PlotProbes(forwards_file_name_list, plot_dir)
            plotter.verbose = True
            plotter.co_param_list = [{
                "filename":os.path.join(base_dir, "closed_orbits_cache"),
                "ref_to_bump_station_mapping":dict([(i,i) for i in range(1001)]),
            },]
            try:
                plotter.load_data()
            except IOError:
                print("IOError trying", for_glob_name)
                raise
            plotter.title_text_1 = str(os.path.split(a_dir)[1])
            plotter.verbose = True
            plotter.plot_phase_spaces()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Done")
