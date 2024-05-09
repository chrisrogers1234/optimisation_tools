import os
import sys
import glob
import json
import matplotlib
import matplotlib.pyplot

class DataHandler(object):
    def __init__(self, file_name_glob, sort_key_list, x_key):
        self.dir_name_glob = file_name_glob
        self.sort_key_list = sort_key_list # distinguish datasets
        self.x_key = x_key # sort each individual dataset
        self.run_summary_fname = "run_summary_list.json" # output from toy model
        self.optimiser_results = "optimiser.json" # output from optimiser
        self.optimiser_keys = ["angle_u", "angle_v", "foil_angle", "beta_x", "alpha_x", "beta_y", "alpha_y"]

    def load_output_list(self):
        dir_name_list = glob.glob(self.dir_name_glob)
        print("Globbing", self.dir_name_glob, "gives", len(dir_name_list), "files")
        self.output_list = []
        for item in dir_name_list: 
            self.output_list += self.load_run_summary(item)
        print("Loaded "+str(len(self.output_list))+" files with keys")
        for key in sorted(self.output_list[0].keys()):
            print(key)
        for key in sorted(self.output_list[0]['config'].keys()):
            print("    config", key)
        self.muggle_data()

    def load_output(self):
        dir_name_list = glob.glob(self.dir_name_glob)
        output_list = []
        for dir_name in dir_name_list:
            output = self.load_run_summary(dir_name)
            if output is None:
                continue
            optimiser_summary = self.load_optimiser_results(dir_name)
            output["optimiser_summary"] = optimiser_summary
            output_list.append(output)
        print("Loaded "+str(len(output_list))+" items with keys")
        for key in sorted(output_list[0].keys()):
            print(key)
        self.output_list = output_list
        self.muggle_data()

    def load_run_summary(self, dir_name):
        file_name = os.path.join(dir_name, self.run_summary_fname)
        try:
            fin = open(file_name)
            output = json.loads(fin.read())
            return output
        except Exception:
            print("Failed to open", file_name)
            return None

    def load_optimiser_results(self, dir_name):
        file_name = os.path.join(dir_name, self.optimiser_results)
        try:
            fin = open(file_name)
            output = json.loads(fin.read())
            return output
        except Exception:
            print("Failed to open", file_name)
            return None

    def is_different(self, key1, key2):
        if len(key1) != len(key2):
            raise ValueError("lists have different length "+str(key1)+" "+str(key2))
        for key1, key2 in zip(key1, key2):
            if abs(key1 - key2) > 1e-9:
                return True
        return False

    def get_key(self, item):
        key = []
        for sort_key in self.sort_key_list:
            if callable(sort_key):
                key.append(sort_key(item))
            elif sort_key in item:
                key.append(item[sort_key])
            else:
                key.append(item['config'][sort_key])
        return key

    def muggle_data(self):
        output = self.output_list
        key = self.get_key(output[0])
        found_keys = []
        for item in output:
            key = self.get_key(item)
            already_found = False
            for test_key in found_keys:
                if not self.is_different(key, test_key):
                    already_found = True
                    break
            if not already_found:
                found_keys.append(key)
        found_keys = sorted(found_keys)

        data_set_list = []
        for key in found_keys:
            print("Found data with", self.sort_key_list, key)
            data_set = []
            for item in output:
                item_key = self.get_key(item)
                if not self.is_different(key, item_key):
                    data_set.append(item)
            data_set = sorted(data_set, key = self.get_key)
            data_set_list.append(data_set)

        self.output_list = data_set_list

class Plotter(object):
    def __init__(self, sort_key_list, sort_name_list, sort_unit_list, x_key):
        self.x_key = x_key
        self.sort_key_list = sort_key_list
        self.colors = ['b', 'g', 'xkcd:slate', 'xkcd:pinkish', 'orange',]# 'r']
        self.xlabel = "Number of turns"
        self.ylabel = "Foil hits"
        self.y_keys = ["mean_foil_hits", "max_foil_hits"] # can also be a list of functions
        self.linestyles = ["dotted", "dashed", "dashdot"]
        self.sort_name = sort_name_list
        self.sort_units = sort_unit_list
        self.file_name = "foil_hits.png"
        self.y_lim = None

    def do_plots(self, data_set_list, do_log, print_data=False):
        try:
            self.do_plots_(data_set_list, do_log, print_data)
        except Exception:
            sys.excepthook(*sys.exc_info())

    def get_value(self, data, key):
        if callable(key):
            return key(data)
        elif key in data:
            return data[key]
        else:
            return data['config'][key]

    def do_plots_(self, data_set_list, do_log, print_data):
        fig = matplotlib.pyplot.figure()
        axes = fig.add_subplot(1, 1, 1, position=[0.15, 0.1, 0.82, 0.85])
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        for i, data_set in enumerate(data_set_list):
            color = self.color(i)
            sort_value = [self.get_value(data_set[0], key) for key in self.sort_key_list]
            x_values = [self.get_value(data, self.x_key) for data in data_set]
            if print_data:
                print(self.x_key, x_values)

            for j, y_key in enumerate(self.y_keys):
                linestyle = self.linestyle(i)
                y_values = [self.get_value(data, y_key) for data in data_set]
                if do_log:
                    func = axes.semilogy
                else:
                    func = axes.plot
                if j == 0:
                    my_label = ""
                    for i, name in enumerate(self.sort_name):
                        my_label += f"{name} {sort_value[i]} {self.sort_units[i]}"
                else:
                    my_label = None
                func(x_values, y_values, color=color, linestyle=linestyle, label=my_label)
                if print_data:
                    print(y_key, y_values)
        if self.y_lim != None:
            axes.set_ylim(self.y_lim)
        axes.legend()
        fig.savefig(self.out_dir+self.file_name)

    def color(self, i):
        i %= len(self.colors) 
        return self.colors[i]

    def linestyle(self, i):
        j = int(i/len(self.colors))
        j %= len(self.linestyles)
        style = self.linestyles[j]
        print(style, j, i)
        return style


def last_turn_hit(data):
    hits_per_turn = data["hits_per_turn"]
    n_lost = data["n_outside_acceptance"]
    print("hits_per_turn", hits_per_turn)
    last_turn = len(hits_per_turn)
    for hits in reversed(hits_per_turn):
        if hits > n_lost:
            break
        last_turn -= 1
    return last_turn*1.0/len(hits_per_turn)

def plots(data_set_list, out_dir, sort_key_list, sort_name_list, sort_unit_list, x_key, n_colours):
    plotter = Plotter(sort_key_list, sort_name_list, sort_unit_list, x_key)
    plotter.colors = plotter.colors[:n_colours]

    plotter.out_dir = out_dir+"/"
    plotter.xlabel = "number of injection turns"

    plotter.ylabel = "Foil hits"
    plotter.y_keys = ["mean_foil_hits", "max_foil_hits"]
    plotter.file_name = "foil_hits.png"
    plotter.do_plots(data_set_list, False)

    plotter.ylabel = "RMS Emittance [$\\mu m$]"
    plotter.y_keys = ["rms_emittance_u", "rms_emittance_v"]
    plotter.y_lim = [0.0, 5.0]
    plotter.file_name = "rms_emittance.png"
    plotter.do_plots(data_set_list, False)

    plotter.ylabel = "99 % Amplitude [$\\mu m$]"
    plotter.y_keys = ["amplitude_u_1e-2", "amplitude_v_1e-2"]
    plotter.y_lim = [1.0, 200.0]
    plotter.file_name = "amplitude_range.png"
    plotter.do_plots(data_set_list, True)

    plotter.ylabel = "$\\sigma$(dp/p)"
    plotter.y_keys = ["rms_dp_over_p", "dp_over_p_1e-2"]
    plotter.y_lim = [0.001, 0.050]
    plotter.file_name = "dp_over_p_rms.png"
    plotter.do_plots(data_set_list, True)

    plotter.ylabel = "Correlation"
    plotter.y_lim = [-1.0, 1.0]
    plotter.y_keys = [lambda x: x["amplitude_u_v_corr"][0][1]]
    plotter.file_name = "amplitude_correlation.png"
    plotter.do_plots(data_set_list, False)

    plotter.ylabel = "Fractional loss"
    plotter.y_lim = [0.0, 0.2]
    plotter.y_keys = [lambda x: x["n_outside_acceptance"]*1.0/x["n_events"] ]
    plotter.file_name = "loss.png"
    plotter.do_plots(data_set_list, False)

    plotter.ylabel = "Fractional loss (longitudinal)"
    plotter.y_lim = [0.0, 0.2]
    plotter.y_keys = [lambda x: x["n_outside_acceptance_long"]*1.0/x["n_events"] ]
    plotter.file_name = "loss_long.png"
    plotter.do_plots(data_set_list, False)

    plotter.ylabel = "Fractional loss (transverse)"
    plotter.y_lim = [0.0, 0.2]
    plotter.y_keys = [lambda x: x["n_outside_acceptance_trans"]*1.0/x["n_events"] ]
    plotter.file_name = "loss_trans.png"
    plotter.do_plots(data_set_list, False)

    return
    # these are only available for optimiser

    plotter.ylabel = "99.9 % range for dp/p"
    plotter.y_keys = ["dp_over_p_1e-3_range"]
    plotter.y_lim = [0.0, 0.03]
    plotter.file_name = "dp_over_p_range.png"
    plotter.do_plots(data_set_list, False)

    plotter.ylabel = "$\\beta_{x,y}$ [m]"
    plotter.y_keys = [
        lambda x: x["optimiser_summary"]["iterations"][0][1][3],
        lambda x: x["optimiser_summary"]["iterations"][0][1][5]
    ] #[lowest_score][inputs][beta_x]
    plotter.y_lim = [0.0, 5.0]
    plotter.file_name = "beta.png"
    plotter.do_plots(data_set_list, False)


    plotter.ylabel = "$\\alpha_{x,y}$ [m]"
    plotter.y_keys = [
        lambda x: x["optimiser_summary"]["iterations"][0][1][4],
        lambda x: x["optimiser_summary"]["iterations"][0][1][6]
    ] #[lowest_score][inputs][beta_x]
    plotter.y_lim = [-4.0, 4.0]
    plotter.file_name = "alpha.png"
    plotter.do_plots(data_set_list, False)


def main():
    dir_base = "output/2023-03-01_baseline/toy_model_painting_v18"
    file_glob = dir_base
    out_dir = dir_base+"/"
    #file_glob = "output/triplet_baseline/single_turn_injection/toy_model_2/*/run_summary.json"
#    sort_key_list = [lambda x: x["config"]["is_correlated"], "amplitude_acceptance"]
#    sort_name_list = ["Corr:", "Acceptance"]
#    sort_unit_list = ["", "$\\mathrm{\\mu}$m"]
    sort_key_list = [lambda x: x["config"]["is_correlated"], "rf_voltage"]
    sort_name_list = ["Corr:", "V$_{rf}$"]
    sort_unit_list = ["", "MV"]
    n_colours = 2
    x_key = "number_pulses"

    data_handler = DataHandler(file_glob, sort_key_list, x_key)
    data_handler.load_output_list()
    #output_list = data_handler.output_list[0:2]+data_handler.output_list[3:]
    #output_list = data_handler.output_list[2:4]
    #out_dir = dir_base+"/foil/"
    output_list = data_handler.output_list
    plots(output_list, out_dir, sort_key_list, sort_name_list, sort_unit_list, x_key, n_colours)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Finished - Press <CR> to close")