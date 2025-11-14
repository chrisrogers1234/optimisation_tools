"""
Plot the fields as a function of time;

Use trajectory of closed orbit from toy model and bump settings from find_bump_parameters
"""
import glob
import json

import matplotlib
import scipy.interpolate
import numpy

import xboa.common

import optimisation_tools.utils.utilities
import optimisation_tools.plotting.plot as plot

class PlotFields:
    def __init__(self, config):
        self.config = config
        self.trajectory = []
        self.magnet_data = []
        self.interpolation = {}
        self.n_injection_turns = -1
        self.n_collapse_turns = -1
        self.tof = 0.0

    def load_co_trajectory(self):
        toy_data = open(self.config["toy_model_run_summary"]).read()
        toy_data = json.loads(toy_data)
        self.trajectory = toy_data["beam_trajectory"]
        end = self.trajectory[-1]
        self.trajectory = [[x-end[i] for i, x in enumerate(point)] for point in self.trajectory]
        self.n_injection_turns = toy_data["config"]["n_injection_turns"]
        self.n_collapse_turns = toy_data["config"]["max_turn"]-self.n_injection_turns
        for i, point in enumerate(self.trajectory[::5]):
            print("    ", i*5, [f"{x:2.6f}" for x in point])

    def get_tof(self, score_list):
        half = 9
        three_quart = 13
        half_tof = [score for score in score_list if score["station"] == half][0]["orbit"][4]
        three_quart_tof = [score for score in score_list if score["station"] == three_quart][0]["orbit"][4]
        tof = half_tof + (three_quart_tof-half_tof)*2
        return tof

    def load_one_magnet_setting(self, filename, verbose):
        with open(filename) as fin:
            for line in fin:
                pass
        magnet_data = json.loads(line)
        bump = self.config["bump_station"]
        co = self.config["co_station"]
        score_list = magnet_data["score_list"]
        bumped_orbit = [score for score in score_list if score["station"] == bump][0]["orbit"]
        closed_orbit = [score for score in score_list if score["station"] == co][0]["orbit"]
        offset = [bumped_orbit[i]-closed_orbit[i] for i in range(6)]
        field_values = magnet_data["parameters"]

        get_value = lambda name: [field["current_value"] for field in field_values if field["name"] == name][0]
        field_values = dict([(name, get_value(name)) for name in self.config["field_keys"]])
        tof = self.get_tof(score_list)
        self.magnet_data.append({"field_values":field_values, "offset":offset, "bumped_orbit":bumped_orbit, "closed_orbit":closed_orbit, "tof":tof})
        if verbose:
            print(f"Score for filename {filename} {magnet_data['score']}")
            for name, value in field_values.items():
                print(f"    {name}: {value}")
            print(f"    Delta: {offset} Bumped: {bumped_orbit} Closed: {closed_orbit} TOF: {tof}")


    def load_magnet_settings(self):
        file_list = sorted(glob.glob(self.config["magnet_data_glob"]))
        for i, filename in enumerate(file_list):
            self.load_one_magnet_setting(filename, i < 1 or i > len(file_list)-2)
            if i == 1 and i <= len(file_list)-2:
                print(f"\n... Loading {len(file_list)} magnet settings ...\n")
        self.tof = numpy.mean([item["tof"] for item in self.magnet_data])

    def setup_interpolations(self):
        field_maps = {}
        for name in self.config["field_keys"]:
            points = [setting["offset"][:2] for setting in self.magnet_data]
            values = [setting["field_values"][name] for setting in self.magnet_data]
            self.interpolation[name] = scipy.interpolate.LinearNDInterpolator(points, values)

    def get_field_values(self, name):
        value_list = []
        # bump
        for i in range(self.n_injection_turns):
            point = self.trajectory[i][:2]
            value = self.interpolation[name](point)
            value_list.append(value)
        end_value = value_list[-1]
        # bump collapse
        for i in range(self.n_collapse_turns+1):
            value = (self.n_collapse_turns-i)/self.n_collapse_turns*end_value
            value_list.append(value)
        return value_list

    def plot_field(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        for name in self.config["field_keys"]:
            value_list = self.get_field_values(name)
            tof_list = [i*self.tof for i in range(len(value_list))]
            axes.plot(tof_list, value_list, label=name)
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Bump field [T]")
        axes.legend()
        axes.set_title(self.config["title"])
        figure.savefig(self.config["output_dir"]+"/field_vs_time.png")


def main():
    output_dir = "output/2023-03-01_baseline/toy_model_painting_v23/corr=True;_t_offset=0.0_dt=200_rf=0.001/"
    config = {
        "output_dir":output_dir,
        "toy_model_run_summary":f"{output_dir}/run_summary_dict.json",
        "magnet_data_glob":"output/2023-03-01_baseline/find_bump_v17/bump=*_by=0.1_bumpp=*/find_bump_parameters_002.out",
        "bump_station":5, # bump orbit station (for magnet settings)
        "co_station":1, # closed orbit station (for magnet settings)
        "field_keys":[f"h bump {i}" for i in range(1, 6)],
        "title":"Correlated",
    }
    plotter = PlotFields(config)
    plotter.load_co_trajectory()
    plotter.load_magnet_settings()
    plotter.setup_interpolations()
    plotter.plot_field()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("Press <CR> to finish")

