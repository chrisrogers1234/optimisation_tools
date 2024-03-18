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

from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix

DecoupledTransferMatrix.det_tolerance = 1

class PlotG4BL(object):
    def __init__(self, run_dir_glob, co_file, cell_length, reference_file, reference_file_format, plot_dir, max_score):
        self.plot_dir = plot_dir
        self.co_data = []
        self.cell_length = cell_length
        self.load_data(run_dir_glob, co_file, reference_file, reference_file_format)
        self.parse_substitutions() # check for variables and sort the list
        self.filter_data(max_score) # if max_score, reject data with data["errors"] > max_score
        self.colors =  ["C"+str(i) for i in range(len(self.co_data))]
        utilities.clear_dir(plot_dir)

    def load_data(self, run_dir_glob, co_file, reference_file, reference_file_format):
        globble = []
        if type(run_dir_glob) == type([]):
            for run_dir in run_dir_glob:
                globble += glob.glob(run_dir)
        else:
            globble = glob.glob(run_dir_glob)
        for a_dir in sorted(globble):
            print(globble)
            new_co_data = []
            reference_data_glob = sorted(glob.glob(os.path.join(a_dir, reference_file))) # output.txt
            for i, file_name in enumerate(reference_data_glob):
                new_co_data.append({"bunch_list":Bunch.new_list_from_read_builtin(reference_file_format, file_name)})
            co_file_name = os.path.join(a_dir, co_file) # closed_orbits_cache
            try:
                fin = open(co_file_name)
                co_data = json.loads(open(co_file_name).read())
                if len(co_data) != len(new_co_data):
                    raise ValueError("Found", len(co_data), "transfer maps and", len(new_co_data), "reference outputs")
                for i, co in enumerate(co_data):
                    new_co_data[i].update(co)
            except OSError:
                print("Failed to load", co_file_name)
            self.co_data += new_co_data

    def _sort_lambda(self, data):
        if "variables" not in data:
            return []
        return [data["variables"][key] for key in self.key_list]

    def filter_data(self, max_score):
        if not max_score:
            return
        co_data_tmp = []
        for data in self.co_data:
            if "errors" not in data:
                continue
            if sum(data["errors"]) < max_score:
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

    def do_plots(self):
        self.figure1 = matplotlib.pyplot.figure()
        self.figure2 = matplotlib.pyplot.figure()
        self.plot_reference_z()
        self.figure1.savefig(self.plot_dir+"/z_vs_p.png")
        self.figure2.savefig(self.plot_dir+"/z_vs_xy.png")
        print("Plotted data with variables", self.key_list)
        max_keys = max([(len(data.keys()), data.keys()) for data in self.co_data])
        print("Top level data keys were", max_keys[1])

    def plot_reference_z(self):
        axes1 = self.figure1.add_subplot(1, 1, 1)
        axes2 = self.figure2.add_subplot(1, 1, 1)
        for i, data in enumerate(self.co_data):
            bunch_list = data["bunch_list"]
            ref_list = []
            for bunch in bunch_list:
                ref_list += bunch.get_hits("event_number", 1)
            print("Found", len(ref_list), "reference hits")

            c = self.colors[i]
            print([hit["p"] for hit in ref_list])
            label = self.get_label(data)
            axes1.plot([hit["z"] for hit in ref_list], [hit["p"] for hit in ref_list], linestyle='dotted', c=c, label=label)
            axes2.plot([hit["z"] for hit in ref_list], [hit["x"] for hit in ref_list], linestyle='dotted', c=c, label=label)
            axes2.plot([hit["z"] for hit in ref_list], [hit["y"] for hit in ref_list], linestyle='dashed', c=c, label=label)

        axes1.set_xlabel("z [mm]")
        axes1.set_ylabel("P [MeV/c]")
        axes1.legend(loc="upper right")
        axes2.legend(loc="upper right")


    key_subs = {
            "__dipole_field__":"B$_{y}$",
            "__momentum__":"p$_{tot}$",
            "__wedge_opening_angle__":"$\\theta_{wedge}$",
            "__energy__":None,
    }
    units_subs = {
            "__dipole_field__":"[T]",
            "__momentum__":"[MeV/c]",
            "__wedge_opening_angle__":"$^\\circ$",
            "__energy__":"",
    }
    beta_limit = 1e4

def main():
    run_dir = "output/musr_cooling_v4/"
    run_dir_glob = [run_dir+"*pz=10_*orbit/"]#, run_dir+"by=0.2_pz=200_r0=*/", run_dir+"by=0.2_pz=220_r0=*/"]
    plot_dir = run_dir+"/plot_momentum/"

    #run_dir_glob = [run_dir+"by=0.05_pz=?00/", run_dir+"by=0.05_pz=60/"]
    #plot_dir = run_dir+"/scan_plots_momentum_restricted/"
    file_name = "tmp/find_closed_orbits/output*.txt"
    co_file_name = "closed_orbits_cache"
    cell_length = 100.0 # full cell length
    file_format = "icool_for009"
    plotter = PlotG4BL(run_dir_glob, co_file_name, cell_length, file_name, file_format, plot_dir, 1e9)
    plotter.beta_limit = 1e4
    plotter.do_plots()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")