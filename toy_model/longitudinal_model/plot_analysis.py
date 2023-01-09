import glob
import os
import json
import matplotlib
import matplotlib.colors
import matplotlib.pyplot


class PlotAnalysis(object):
    def __init__(self):
        self.data = []
        self.plot_dir = "./"

    def load_data(self, output_directory_list):
        self.data = []
        for a_dir in output_directory_list:
            with open(os.path.join(a_dir, "analysis.dat")) as fin:
                my_data = json.loads(fin.read())
            with open(os.path.join(a_dir, "config.json")) as fin:
                config = json.loads(fin.read())
                my_data.update(config)
            self.data.append(my_data)

    def plot(self, variable, fmt_string):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        my_cm = matplotlib.pyplot.cm.viridis
        colors = (my_cm(i/len(self.data)) for i in range(len(self.data)))
        for my_data in self.data:
            value = format(my_data[variable], fmt_string)
            label = self.label(variable)+f" {value}"
            axes.plot(my_data["t_list"], my_data["v_p2p_list"], label=label, color=next(colors))
        axes.set_xlabel("t [ns]")
        axes.set_ylabel("Peak-to-peak voltage [AU]")
        axes.legend()
        figure.savefig(os.path.join(self.plot_dir, f"voltages-{variable}.png"))

    def label(self, name):
        return name



def main():
    output_directory_list = glob.glob(f"output/kurns_v4/constant_bucket_*/")
    for a_dir in output_directory_list:
        print("Plotting individual", a_dir)
        model = PlotAnalysis()
        model.plot_dir = a_dir
        try:
            model.load_data([a_dir])
            model.plot("output_dir", "")
        except FileNotFoundError:
            continue


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
