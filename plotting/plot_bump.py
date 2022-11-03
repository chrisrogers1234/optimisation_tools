import glob
import os
import json
import math
import sys

import xboa.common as common
import matplotlib
import matplotlib.pyplot
import ROOT

from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
import optimisation_tools.utils.utilities as utilities


class PlotBump(object):
    def __init__(self, plot_dir):
        self.plot_dir = plot_dir
        self.number_of_phases = 1
        self.target_probe = 0
        self.flip_vertical = False
        self.file_list = None
        self.co_files = None
        self.tm_dict = {}
        self.data = []
        self.target_field = ""
        self.score_cutoff = None
        self.text_str = ""

    def setup(self):
        utilities.clear_dir(self.plot_dir)
        self.load_files()
        for co_params in self.co_files:
            self.load_closed_orbits(co_params)
        self.add_decoupled()

    def sort_key(self, filename):
        return float(filename.split("_r_")[1].split("_theta")[0])

    def sort_key(self, filename):
        return float(filename.split("_r0_")[1].split("/")[0])

    def sort_key(self, filename):
        words = filename.split("=")[1:]
        words[0] = float(words[0].split("_")[0])
        words[1] = float(words[1].split("/")[0])
        return [words[0], words[1]]

    def load_files(self):
        index = 0
        file_list = []
        for file_glob in self.file_list:
            for file_name in glob.glob(file_glob):
                file_list.append(file_name)
        file_list = sorted(file_list, key=self.sort_key)
        for file_name in file_list:
                score, n_iterations = [], []
                index += 1
                if index % self.number_of_phases != 0:
                    score.append(self.load_one_file(file_name)[0]["score"])
                    n_iterations.append(self.load_one_file(file_name)[0]["n_iterations"])
                    continue
                self.data += self.load_one_file(file_name)
                self.data[-1]["score"] = score+[self.data[-1]["score"]]
                self.data[-1]["n_iterations"] = n_iterations+[self.data[-1]["n_iterations"]]
        bump_fields = dict([(field, 0) for field in self.data[0]["bump_fields"]])
        self.data = [{ # reserve space for reference tracks
                "tracking":[],
                "bump_fields":bump_fields,
                "score":[1e-9]*self.number_of_phases,
                "n_iterations":[1]*self.number_of_phases,
                "target_hit":None,
                "target_orbit":None,
                "subs":None,
                "optimisation_stage":None,
        }]+self.data

    def load_one_file(self, file_name):
        fin = open(file_name)
        print("Loading", file_name, end=' ')
        sys.stdout.flush()
        data_out = None
        n_iterations = 0
        score_list = []
        for line in fin.readlines():
            data_tmp = json.loads(line)
            score_list.append(data_tmp['score'])
            if data_out == None or data_tmp['score'] < data_out['score']:
                data_out = data_tmp
            n_iterations += 1
        if len(data_out) == 0:
            print(" empty file, abort")
            sys.exit(1)
        print("... loaded", n_iterations, "lines with",
              "worst score", format(max(score_list), "8.4g"), 
              "best score", format(data_out['score'], "8.4g"))
        if self.flip_vertical:
            for hit in data_out["tracking"]:
                hit[3] *= -1
                hit[4] *= -1
        data_out['n_iterations'] = n_iterations
        return [data_out]

    def load_closed_orbits(self, co_params):
        fin = open(co_params["filename"])
        co = json.loads(fin.readline())[0]

        ref_to_bump_station_mapping = co_params["ref_to_bump_station_mapping"]
        existing_stations = set([track[0] for track in self.data[0]["tracking"]])
        for hit in co["ref_track"]:
            try:
                bump_station = ref_to_bump_station_mapping[hit["station"]]
            except KeyError: # if hit station not in the list, we ignore it
                continue
            if bump_station not in existing_stations:
                track = [bump_station, hit["x"], hit["px"], hit["y"], hit["py"]]
                self.data[0]["tracking"].append(track)
                existing_stations.add(bump_station)
        tm = [row[1:5] for row in co["tm"]]
        tm = DecoupledTransferMatrix(tm, True)
        for key, var in ref_to_bump_station_mapping.items():
            self.tm_dict[var] = tm

    def add_decoupled(self):
        ref_dict = dict([(item[0], item[0:5]) for item in self.data[0]["tracking"]])
        for j, item in enumerate(self.data):
            for i, track in enumerate(item["tracking"]):
                try:
                    ref = ref_dict[track[0]]
                    tm = self.tm_dict[track[0]]
                except KeyError:
                    item["tracking"][i] = track+[0.]*8
                    continue
                coupled = [track[k] - ref[k] for k in range(1, 5)]
                decoupled = tm.decoupled(coupled).tolist()
                action_angle = tm.coupled_to_action_angle(coupled)
                item["tracking"][i] = track+decoupled+action_angle
                if j < 3:
                    print("hit:", i, "ev:", j, "track:", item["tracking"][i])

    def print_field_substitutions(self):
        #bump_data = [bump["bump_fields"] for bump in self.data]
        for bump in self.data:
            bump_data = bump["bump_fields"]
            if bump["target_orbit"] != None:
                print("Score:", bump["score"], "Bump at probe", str(self.target_probe), "target", bump["target_orbit"][str(self.target_probe)]) 
            print(json.dumps(bump_data, indent=2))

    def plot_all_fields_2d(self, var_1, var_2):
        field_name_list = sorted(self.data[0]["bump_fields"])
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        index = 1
        for i, name in enumerate(field_name_list):
            axes = figure.add_subplot(2, 5, i+1)
            utilities.setup_large_figure(axes)
            z_lambda = lambda bump: bump["bump_fields"][name]
            scatter = self.plot_2d(var_1, var_2, z_lambda, name, False, False, axes)
            figure.colorbar(scatter, ax=[axes])
            index += 1

    def plot_phased_var_2d(self, var_1, var_2, var_3):
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        index = 0
        for var_z in var_3:
            for phase_index in range(self.number_of_phases):
                index += 1
                axes = figure.add_subplot(len(var_3), self.number_of_phases, index)
                utilities.setup_large_figure(axes)
                z_lambda = lambda bump: bump[var_z][phase_index]
                name = var_z+" for phase "+str(phase_index+1)
                scatter = self.plot_2d(var_1, var_2, z_lambda, name, True, False, axes)
                figure.colorbar(scatter, ax=[axes])
        fig_name = "phased_plot_"+str(var_1)+"_"+str(var_2)+"_station-"+str(self.target_probe)+".png"
        figure.savefig(fig_name)


    def plot_event_display(self, z_lambda):
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        axes = [
            figure.add_subplot(2, 3, 1,  position=[0.1, 0.55, 0.43, 0.35]),
            figure.add_subplot(2, 6, 7,  position=[0.1, 0.10, 0.16, 0.35]),
            figure.add_subplot(2, 6, 8,  position=[0.38, 0.10, 0.16, 0.35]),
            #figure.add_subplot(2, 3, 2,  position=[0.38, 0.55, 0.26, 0.35]),
            #figure.add_subplot(2, 6, 9,  position=[0.38, 0.10, 0.10, 0.35]),
            #figure.add_subplot(2, 6, 10, position=[0.54, 0.10, 0.10, 0.35]),
            figure.add_subplot(2, 3, 3,  position=[0.65, 0.55, 0.43, 0.35]),
            figure.add_subplot(2, 6, 11, position=[0.65, 0.10, 0.16, 0.35]),
            figure.add_subplot(2, 6, 12, position=[0.88, 0.10, 0.16, 0.35]),
        ]
        scatter = self.plot_2d(1, 3, z_lambda, "", False, True, axes[0])
        self.plot_2d(1, 2, z_lambda, "", False, True, axes[1])
        self.plot_2d(3, 4, z_lambda, "", False, True, axes[2])
        #self.plot_2d(5, 7, z_lambda, "", False, True, axes[3])
        #self.plot_2d(5, 6, z_lambda, "", False, True, axes[4])
        #self.plot_2d(7, 8, z_lambda, "", False, True, axes[5])
        self.plot_2d(10, 12, z_lambda, "", False, True, axes[3])
        self.plot_2d(9, 10, z_lambda, "", False, True, axes[4])
        self.plot_2d(11, 12, z_lambda, "", False, True, axes[5])
        self.text(axes[0])
        self.text(axes[3])
        title = self.target_field.replace("__", "").replace("_", " ")
        title += " [T] at station "+str(self.target_probe)
        figure.suptitle(title)
        figure.colorbar(scatter, ax=axes)
        fig_name = "event_display_"+self.target_field.replace("__", "")+"_station-"+str(self.target_probe)+".png"
        figure.savefig(os.path.join(self.plot_dir, fig_name))


    def plot_2d(self, foil_var1, foil_var2, z_lambda, plot_title, logbar, labels, axes):
        utilities.setup_large_figure(axes)
        name = plot_title.replace("__", "")
        name = name.replace("_", " ")
        x_data, y_data, z_data = [], [], []
        for bump in self.data:
            for hit in bump["tracking"]:
                if hit[0] == self.target_probe:
                    x_data.append(hit[foil_var1])
                    y_data.append(hit[foil_var2])
                    break
            if len(x_data) != len(z_data): # meaning we found target probe
                z = z_lambda(bump)
                if z:
                    z_data.append(z)
                else:
                    del x_data[-1]
                    del y_data[-1]

        n_points = len(x_data)
        if axes == None:
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1, 1, 1)
        norm = None
        if logbar:
            norm = matplotlib.colors.LogNorm()
        point_size = 10 #max(2, 10/n_points**0.5)
        scatter = axes.scatter(x_data, y_data, c=z_data, s=point_size, norm=norm)
        axes.set_title(name)
        suptitle = self.var[foil_var1]+" vs "+self.var[foil_var2]
        axes.get_figure().suptitle(suptitle)
        for char1, char2 in ("[", ""), ("]", ""), (" ", "_"), ("/", ""):
            suptitle = suptitle.replace(char1, char2)
        figname = suptitle.replace(" ", "_").replace("[", "").replace("]", "")
        if labels:
            axes.set_xlabel(self.var[foil_var1])
            axes.set_ylabel(self.var[foil_var2])

        axes.get_figure().savefig(self.plot_dir+"/"+suptitle.replace(" ", "_")+".png")
        return scatter

    def plot_fields(self, x_axis):
        bump_fields = {}
        x_values = []
        for bump in self.data:
            if self.score_cutoff and min(bump["score"]) > self.score_cutoff:
                continue
            for key in bump["bump_fields"]:
                if key not in bump_fields:
                    bump_fields[key] = []
                bump_fields[key].append(bump["bump_fields"][key])

            if type(x_axis) == type(0):
                x_values.append(None)
                for hit in bump["tracking"]:
                    if hit[0] == self.target_probe:
                        x_values[-1] = hit[x_axis]
                        print("x", hit[1], "y", hit[3], "score:", bump["score"], bump["target_orbit"][str(self.target_probe)], "vs", hit[0:5])
                        break

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        if type(x_axis) == type(""):
            x_name = x_axis.replace("__", "").replace("_", " ").replace("field", "")
            axes.set_xlabel(x_name+" [T]")
            x_values = bump_fields[x_axis]
        elif type(x_axis) == type(0):
            axes.set_xlabel(self.var[x_axis])
        else:
            axes.set_xlabel("Setting number")
        axes.set_ylabel("Bump field [T]")
        for key in sorted(bump_fields.keys()):
            name = key.replace("__", "").replace("_", " ").replace("field", "")
            if "h bump" in name:
                style = "dotted"
            else:
                style = "dashed"
            axes.plot(x_values, bump_fields[key], label=name, linestyle=style, marker="o")
        axis_xrange = list(axes.get_xlim())
        axis_xrange[1] += (axis_xrange[1]-axis_xrange[0])*0.5
        axes.set_xlim(axis_xrange)
        axes.legend()
        self.text(axes)
        figure.savefig(self.plot_dir+"/bump_fields_"+str(x_axis)+".png")

    def angle_plot(self, bump_1, bump_2, y_var, axes, scale_factor=1.0):
        x_axis, y_axis = [], []
        for item in self.data:
            bump_field_1 = item["bump_fields"][bump_1]
            bump_field_2 = item["bump_fields"][bump_2]
            bump_angle = math.atan2(bump_field_1, bump_field_2)
            x_axis.append(math.degrees(bump_angle))
            y_axis.append(0)
            for hit in item["tracking"]:
                if hit[0] == self.target_probe:
                    y_axis[-1] = hit[y_var]*scale_factor
                    break
        point_size = max(1, 10/len(x_axis)**0.5)
        label = self.var[y_var]
        if scale_factor != 1.0:
            label =str(scale_factor)+label
        axes.scatter(x_axis, y_axis, s=point_size, label=label)
        axes.set_xlabel("Kick angle [$^\\circ$]")
        axes.set_ylabel(self.var[y_var])
        self.text(axes)
        bump_1_text = bump_1.replace("__", "").replace("_", "").replace("bump", "").replace("field", "")
        bump_2_text = bump_2.replace("__", "").replace("_", " ").replace("bump", "").replace("field", "")
        axes.set_xlabel("$\\mathrm{atan}\\left(\\frac{B("+bump_1_text+")}{B("+bump_2_text+")}\\right)$ [$^\\circ$]")

    def plot_lambda(self, bump):
        if not self.score_cutoff or bump["score"][0] < self.score_cutoff:
            return bump["bump_fields"][self.target_field]

    def text(self, axes):
        if self.text_str:
            axes.text(0.9, 1.01, self.text_str, ha='right', transform=axes.transAxes)


    root_objects = []
    var = {
        0:"station",
        1:"Radial position [mm]", 2:"Radial momentum [MeV/c]",
        3:"Height [mm]", 4:"Vertical momentum [MeV/c]",
        5:"U", 6:"UP", 7:"V", 8:"VP",
        9:"$\\phi_u$", 10:"$A_{u}$",
        11:"$\\phi_v$", 12:"$A_{v}$"
    }

def main():
    by = "0.20"
    root_dir = "output/2022-07-01_baseline/bump_quest_v3"
    file_list = glob.glob(root_dir+"/find_bump_r0=-*by="+by+"/find_bump_*_002.out")
    plot_dir = root_dir+"/plot_bump_by="+by
    DecoupledTransferMatrix.det_tolerance = 1.0

    plotter = PlotBump(plot_dir)
    plotter.text_str = root_dir.split("_")[-1]
    plotter.score_cutoff = 1e9
    plotter.number_of_phases = 1
    plotter.flip_vertical = False
    plotter.file_list = file_list
    plotter.co_files = []
    plotter.setup()

    plotter.target_probe = 4
    plotter.target_field = "__h_bump_2_field__"
    plotter.plot_fields(1)
    plotter.plot_fields(3)
    plotter.plot_phased_var_2d(1, 3, ["score", "n_iterations"])
    for probe in [1, 2, 3, 0, 4, 5, 6, 7]:
        plotter.target_probe = probe
        try:
            plotter.plot_event_display(plotter.plot_lambda)
        except Exception:
            print("Failed at station", probe)
    plotter.target_probe = 4
    plotter.print_field_substitutions()

    plotter.target_probe = 2
    figure = matplotlib.pyplot.figure(figsize=(20,10))
    #for subplot, var in [(1, 10), (2, 12)]:
    axes = figure.add_subplot(1, 1, 1)
    #plotter.angle_plot("__v_bump_1_field__", "__h_bump_1_field__", 10, axes, 10)
    #plotter.angle_plot("__v_bump_1_field__", "__h_bump_1_field__", 12, axes, 1)
    axes.legend()
    figure.savefig(os.path.join(plot_dir, "kick_angle_to_field_angle.png"))
    print("Used plot_dir", plot_dir)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Done - Press <CR> to end")

