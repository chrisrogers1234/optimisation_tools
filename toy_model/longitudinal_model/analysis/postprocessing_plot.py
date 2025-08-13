import glob
import json

import matplotlib
import matplotlib.pyplot

class PostprocessingPlotter:
    def __init__(self):
        self.folder_glob = []
        self.config_filename = ""
        self.postproc_filename = ""
        self.data_list = None
        self.output_dir = ""

    def load_json(self, filename):
        a_str = open(filename).read()
        a_json = json.loads(a_str)
        return a_json

    def load_data(self):
        folder_list = []
        for a_glob in self.folder_glob:
            folder_list += glob.glob(a_glob)
        self.data_list = []
        for folder in sorted(folder_list):
            item = {}
            try:
                item["config"] = self.load_json(f"{folder}/{self.config_filename}")
                item["postproc"] = self.load_json(f"{folder}/{self.postproc_filename}")
                self.data_list.append(item)
            except FileNotFoundError:
                print(f"Failed to load {folder}")
        print("Loaded data")
        for item in self.data_list:
            print("   ", item["config"]["data"]["file_number"])

    def plot(self):
        self.plot_peak_voltage()
        self.plot_derivative_voltage()

    def plot_peak_voltage(self):
        x_list = []
        y_list = []
        for item in self.data_list:
            t_list = [prog_item["time"] for prog_item in item["config"]["rf"]["programme"]]
            if abs(t_list[2]-t_list[1]-8e6) > 1e5:
                continue
            v_list = [prog_item["voltage"] for prog_item in item["config"]["rf"]["programme"]]
            x_list.append(max(v_list))
            y_list.append(item["postproc"]["beam_capture"]["max_value"])
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.scatter(x_list, y_list, s=2)
        axes.set_xlabel("Logbook maximum RF voltage [kV]")
        axes.set_ylabel("Peak integrated BPM turn voltage [AU]")
        figure.savefig(f"{self.output_dir}/integrated_particles_vs_voltage.png")

    def plot_derivative_voltage(self):
        x_list = []
        y_list_first = []
        y_list_ten_pc = []
        for item in self.data_list:
            t_list = [prog_item["time"] for prog_item in item["config"]["rf"]["programme"]]
            if abs(t_list[2]-t_list[1]-8e6) > 1e5:
                continue
            v_list = [prog_item["voltage"] for prog_item in item["config"]["rf"]["programme"]]
            x_list.append(max(v_list))
            y_list_first.append(item["postproc"]["beam_capture"]["first_minimum_rf_voltage"])
            y_list_ten_pc.append(item["postproc"]["beam_capture"]["10_pc_voltage"])
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.scatter(x_list, y_list_first, s=2, label="RF voltage of first minimum")
        axes.scatter(x_list, y_list_ten_pc, s=2, label="RF voltage at 10 % of peak")
        axes.set_xlabel("Logbook maximum RF voltage [kV]")
        axes.set_ylabel("RF voltage at 'flat top' [AU]")
        axes.legend()
        figure.savefig(f"{self.output_dir}/flat_top_rf_voltage.png")



def main():
    plotter = PostprocessingPlotter()
    plotter.output_dir = "output/2024-12-21_v6/"
    plotter.folder_glob = [f"{plotter.output_dir}/job_0*"]
    plotter.config_filename = "config.json"
    plotter.postproc_filename = "analysis_output.json"
    plotter.load_data()
    plotter.plot()


if __name__ == "__main__":
    main()
