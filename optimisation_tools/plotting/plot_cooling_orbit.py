
import math
import sys
import glob
import json

import numpy
import matplotlib
import matplotlib.pyplot
import optimisation_tools.utils.search_json
import optimisation_tools.utils.decoupled_transfer_matrix
optimisation_tools.utils.decoupled_transfer_matrix.DecoupledTransferMatrix.det_tolerance = 1.0

class PlotOrbit:
    def __init__(self):
        filename_glob = ""
        self.score_patter = ""
        self.p_pattern = "*.$p_*"
        self.x_orbit_pattern = "*.$beam_x_*"
        self.y_orbit_pattern = "*.$beam_y_*"
        self.x_trajectory_pattern = "*.optimisation.Closed orbit on momentum 190.score.sum_square_score_list(*)"
        self.tm_pattern = "*.$tm_*"
        self.score_pattern = "*.optimisation.Closed orbit on momentum *.score_result"
        self.score_max = 10.0
        self.data = {}
        self.plot_dir = ""
        self.x_values = []
        self.y_values = []
        self.p_values = []
        self.tm_values = []
        self.score_values = []
        self.beta = []
        self.tune = []
        self.nu1, self.nu2 = [], []

    def load_data(self):
        filenames = glob.glob(self.filename_glob)
        for fname in filenames:
            with open(fname) as f:
                self.data[fname] = json.loads(f.read())
        self.x_values, paths = self.get_sorted_values(self.x_orbit_pattern)
        self.y_values, paths = self.get_sorted_values(self.y_orbit_pattern)
        self.p_values, paths = self.get_sorted_values(self.p_pattern)
        self.tm_values, paths = self.get_sorted_values(self.tm_pattern)
        self.score_values, paths = self.get_sorted_values(self.score_pattern)
        self.cuts()
        self.tm_analysis()

    def tm_analysis(self):
        DTM = optimisation_tools.utils.decoupled_transfer_matrix.DecoupledTransferMatrix
        for i, tm in enumerate(self.tm_values):
            tm = numpy.array(tm)
            tm_s = DTM.simplectify(tm)
            self.nu1.append(0.0)
            self.nu2.append(0.0)
            self.beta.append(0.0)
            try:
                dm = DTM(tm_s, normalise=True)
                self.nu1[-1] = dm.get_phase_advance(0)/math.pi/2.0
                self.nu2[-1] = dm.get_phase_advance(1)/math.pi/2.0
            except Exception:
                sys.excepthook(*sys.exc_info())
            print(tm,"\n")
            print(tm_s, "\n")
            print(DTM.simplecticity(tm), "\n")
            print(DTM.simplecticity(tm_s), "\n")
            print("Phase advance:", self.nu1[i], self.nu2[i])
            print("p:", self.p_values[i], "score:", self.score_values[i], "beta:", self.beta[i])
            print("==============================")


    def cuts(self):
        score_cut = lambda x: x < self.score_max
        self.x_values = self.one_cut(self.x_values, self.score_values, score_cut)
        self.y_values = self.one_cut(self.y_values, self.score_values, score_cut)
        self.p_values = self.one_cut(self.p_values, self.score_values, score_cut)
        self.tm_values = self.one_cut(self.tm_values, self.score_values, score_cut)
        self.score_values = self.one_cut(self.score_values, self.score_values, score_cut)

    def one_cut(self, values, cut_values, cut_lambda):
        values = [v for i, v in enumerate(values) if cut_lambda(cut_values[i])]
        return values

    def get_sorted_values(self, pattern):
        values, paths = optimisation_tools.utils.search_json.search_json(self.data, pattern)
        if len(values) == 0:
            return [], []
        a_sort = sorted(zip(paths, values))
        paths, values = zip(*a_sort)
        return values, paths

    def plots(self):
        self.plot_orbit()
        self.plot_tm()

    def plot_orbit(self):

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        axes.plot(self.x_values, self.y_values)
        axes.set_xlabel("x [mm]")
        axes.set_ylabel("y [mm]")
        figure.savefig("orbit_x_vs_y.png")

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        axes.plot(self.p_values, self.x_values, label = "x [mm]")
        axes.plot(self.p_values, self.y_values, label = "y [mm]")
        axes.set_xlabel("P [MeV/c]")
        axes.set_ylabel("Position [mm]")
        axes.legend()
        figure.savefig("orbit_p_vs_xy.png")

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        axes.plot(self.p_values, self.score_values)
        axes.set_xlabel("P [MeV/c]")
        axes.set_ylabel("Score")
        figure.savefig("orbit_p_vs_score.png")

    def plot_tm(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        axes.plot(self.p_values, self.nu1, label="$\\nu_1$")
        axes.plot(self.p_values, self.nu2, label="$\\nu_2$")
        axes.set_xlabel("P [MeV/c]")
        axes.set_ylabel("$\\nu$")
        figure.savefig("tm_p_vs_nu.png")



def main():
    #Test().test_search_json()
    plotter = PlotOrbit()
    plotter.filename_glob = "output/v075/name=cooling;_gap=1000;_pz=200;_rf_gap_phase_delta=0;_rf_gap_scale=1/optimisation_closed_orbit"
    plotter.x_label = "p [MeV/c]"
    plotter.y_label = "y [mm]"
    plotter.load_data()
    values, paths = optimisation_tools.utils.search_json.search_json(plotter.data, plotter.x_trajectory_pattern)
    print(values)
    print(json.dumps(paths, indent=2))
    plotter.plots()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")

