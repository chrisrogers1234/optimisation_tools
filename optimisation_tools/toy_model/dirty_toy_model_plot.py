import json
import os
import matplotlib.pyplot

class Plotter:
    def __init__(self, file):
        self.file = file
        self.outdir = os.path.split(file)[0]
        self.data = None

    def load_data(self):
        data_string = open(self.file).read()
        self.data = json.loads(data_string)

    def plot_data(self):
        print(self.data[0].keys())
        print(self.data[0]["config"].keys())
        anticorrelated_data = [item for item in self.data if "anticorrelated" in item["config"]["study"]]
        correlated_data = [item for item in self.data if "correlated" in item["config"]["study"] and "anti" not in item["config"]["study"]]
        print(len(anticorrelated_data), [item["config"]["study"] for item in anticorrelated_data])
        print(len(correlated_data), [item["config"]["study"] for item in correlated_data])
        anticorrelated_data_0 = [item for item in anticorrelated_data if item["config"]["version"] < 6]
        anticorrelated_data_1 = [item for item in anticorrelated_data if item["config"]["version"] >= 6]
        correlated_data_0 = [item for item in correlated_data if item["config"]["version"] < 6]
        correlated_data_1 = [item for item in correlated_data if item["config"]["version"] >= 6]
        mean_hits_fig = matplotlib.pyplot.figure("mean hits")
        mean_hits_axes = mean_hits_fig.add_subplot(1, 1, 1)
        dp_p_fig = matplotlib.pyplot.figure("dp over p")
        dp_p_axes = dp_p_fig.add_subplot(1, 1, 1)
        n_accepted_fig = matplotlib.pyplot.figure("n accepted")
        n_accepted_axes = n_accepted_fig.add_subplot(1, 1, 1)

        for name, data in [
            ("anticorrelated y", anticorrelated_data_0),
            ("anticorrelated y'", anticorrelated_data_1),
            ("correlated y", correlated_data_0),
            ("correlated y'", correlated_data_1),
            ]:
            n_pulses_list = [item["config"]["number_pulses"] for item in data]
            mean_hits_list = [item["mean_foil_hits"] for item in data]
            dp_p_list = [item["dp_over_p_1e-2"] for item in data]
            n_accepted_list = [item["n_outside_acceptance"]/(item["config"]["number_per_pulse"]*item["config"]["number_pulses"]) for item in data]
            mean_hits_axes.plot(n_pulses_list, mean_hits_list, label = name)
            dp_p_axes.plot(n_pulses_list, dp_p_list, label = name)
            print(n_accepted_list)
            n_accepted_axes.plot(n_pulses_list, n_accepted_list, label = name)

        mean_hits_axes.set_xlabel("Number of injection turns")
        mean_hits_axes.set_ylabel("Mean number of foil hits (including first)")
        mean_hits_axes.legend()
        mean_hits_fig.savefig(f"{self.outdir}/mean_hits.png")

        dp_p_axes.set_xlabel("Number of injection turns")
        dp_p_axes.set_ylabel("99 % dp/p")
        dp_p_axes.legend()
        dp_p_fig.savefig(f"{self.outdir}/dp_p.png")

        n_accepted_axes.set_xlabel("Number of injection turns")
        n_accepted_axes.set_ylabel("Fraction of beam lost")
        n_accepted_axes.legend()
        n_accepted_fig.savefig(f"{self.outdir}/n_accepted.png")

def main():
    my_plotter = Plotter("output/2023-03-01_baseline/baseline/toy_model_painting/run_summary_list.json")
    my_plotter.load_data()
    my_plotter.plot_data()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("press <CR> to finish")