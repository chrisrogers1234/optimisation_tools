import glob
import os
import json
import math
import sys

import numpy
import ROOT

import xboa.common as common
import matplotlib
import matplotlib.pyplot

from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
import optimisation_tools.utils.utilities as utilities

class QuadFinder(object):
    def __init__(self, tm, data_def):
        self.k1 = data_def["k1"]
        self.k2 = data_def["k2"]
        self.ld1 = data_def["l_d1"] # drift length
        self.ld2 = data_def["l_d2"] # drift length
        self.lq = data_def["l_q"] # quad length
        self.target_nu_x = data_def["nu_x"]
        self.target_nu_y = data_def["nu_y"]
        self.tm = numpy.array(tm)        
        self.key = data_def["key"]
        self.iteration = 0
        self.minuit = ROOT.TMinuit(2)

    def quad_tm(self, k):
        l = self.lq
        if k > 0:
            quad_tm = [[math.cos(l*k**0.5), math.sin(l*k**0.5)/k**0.5],
                       [-k**0.5*math.sin(l*k**0.5), math.cos(l*k**0.5)]]
        elif k == 0:
            quad_tm = [[1, self.lq],
                       [0, 1]]
        elif k < 0:
            k *= -1
            print(k, l)
            print(math.cosh(l*k**0.5))
            quad_tm = [[math.cosh(l*k**0.5), math.sinh(l*k**0.5)/k**0.5],
                       [-(k**0.5)*math.sinh(l*k**0.5), math.cosh(l*k**0.5)]]
        return numpy.array(quad_tm)

    def drift_tm(self, l):
        drift_tm = [[1, l],
                    [0, 1]]
        return numpy.array(drift_tm)

    def quad_thin_tm(self, k):
        quad_tm = [[1, 0],
                   [k, 1]]
        return numpy.array(quad_tm)

    def get_matrix(self, sign):
        tm = numpy.array([[1, 0], [0, 1]])
        for matrix in [self.quad_thin_tm(sign*self.k1), self.drift_tm(self.ld1), self.quad_thin_tm(sign*self.k2), self.drift_tm(self.ld2)]:
            tm = numpy.dot(tm, matrix)
        return tm

    def get_trace(self, tm, sign):
        tm = numpy.dot(tm, self.get_matrix(sign))
        trace = tm[0,0]+tm[1,1]
        print(tm)
        return trace

    def update_from_minuit(self):
        value = ROOT.Double()
        error = ROOT.Double()
        self.minuit.GetParameter(0, value, error)
        self.k1 = float(value)
        self.minuit.GetParameter(1, value, error)
        self.k2 = float(value)

    def score(self):
        self.iteration += 1
        tm_x = self.tm[0:2, 0:2]
        tm_y = self.tm[2:4, 2:4]
        print("k", self.k1, self.k2)
        print("x tm")
        tr_x = self.get_trace(tm_x, +1)
        print("y tm")
        tr_y = self.get_trace(tm_y, -1)
        score_x = tr_x/2 - math.cos(2*math.pi*self.target_nu_x)
        score_y = tr_y/2 - math.cos(2*math.pi*self.target_nu_y)
        score = score_y**2+score_x**2
        print("X", tr_x/2, math.cos(2*math.pi*self.target_nu_x))
        print("Y", tr_y/2, math.cos(2*math.pi*self.target_nu_y))
        print("Score:", score, "on iteration", self.iteration)
        return score

    def minuit_function(self, nvar=None, parameters=None, score=[0], jacobian=None, err=None):
        self.update_from_minuit()
        score[0] = self.score()
        return score[0]

    def get_tune(self, tm):
        phase_advance = math.acos((tm[0, 0]+tm[1, 1])/2)
        return phase_advance/2.0/math.pi

    def solve_analytically(self):
        nu_sim_x = self.get_tune(self.tm[0:2, 0:2])
        nu_sim_y = self.get_tune(self.tm[2:4, 2:4])
        nu_req_x = self.target_nu_x-nu_sim_x
        nu_req_y = self.target_nu_y-nu_sim_y
        tx = 2*math.cos(2*math.pi*nu_req_x)
        ty = 2*math.cos(2*math.pi*nu_req_y)
        a = -1
        b = (tx-ty)/2/(self.ld1+self.ld2)
        c = (4-tx-ty)/2/self.ld1/self.ld2
        self.k1 = (-b + (b**2-4*a*c)**0.5)/2/a
        self.k2 = b-self.k1
        tm_x = self.get_matrix(+1)
        tm_y = self.get_matrix(-1)
        print(tm_x, "trace:", tm_x[0,0]+tm_x[1,1], "tx:", tx)
        print(tm_y, "trace:", tm_y[0,0]+tm_y[1,1], "ty:", ty)
        tm_x = numpy.dot(self.tm[0:2, 0:2], self.get_matrix(+1))
        if self.key == "q1":
            return 1/self.k1
        elif self.key == "q2":
            return 1/self.k2
        else:
            raise KeyError(str(self.key))

    def solve_by_hand(self):
        k1, k2 = None, None
        while not k1:
            x = input("Enter k1: ")
            try:
                k1 = float(x)
            except ValueError:
                pass
        while not k2:
            x = input("Enter k2: ")
            try:
                k2 = float(x)
            except ValueError:
                pass
        self.k1 = k1
        self.k2 = k2
        self.score()

    def solve(self):
        self.minuit = ROOT.TMinuit(2)
        self.minuit.DefineParameter(0, "k1", self.k1, 1e-10, 0, 0)
        self.minuit.DefineParameter(1, "k2", self.k2, 1e-10, 0, 0)
        self.minuit.SetFCN(self.minuit_function)
        self.iteration = 0
        self.minuit_function()
        self.minuit.Command("SIMPLEX 1000 1e-6")
        if self.key == "q1":
            return 1/self.k1
        return 1/self.k2

def hack_tune(score, data_def):
    print("WARNING - HACKED TUNE")
    if abs(score) < 1:
        hack_tunes = [0.2131, 0.2119]
        score = 1-score+hack_tunes[data_def["key_list"][1]]*8
        while score > 1:
            score -= 1
    else:
        score = abs(score)
    return score

class PlotFindBumpParameters(object):
    def __init__(self, plot_dir, file_glob):
        self.plot_dir = plot_dir
        self.file_glob = file_glob
        self.data = []

    def setup(self):
        utilities.clear_dir(self.plot_dir)
        self.build_data()
        self.load_subs()
        self.load_bumps()

    def build_data(self):
        file_list = sorted(glob.glob(self.file_glob))
        if not len(file_list):
            raise ValueError(f"No files in {self.file_glob}")
        old_dir, new_dir = None, None
        for file_name in file_list:
            old_dir = new_dir
            new_dir = os.path.split(file_name)[0]
            if new_dir != old_dir:
                new_item = {
                    "dir":new_dir,
                    "file_list":[file_name],
                    "subs":{},
                    "find_bump_parameters":[]
                }
                self.data.append(new_item)
            else:
                self.data[-1]["file_list"].append(file_name)

    def load_bumps(self):
        for item in self.data:
            for a_file in item["file_list"]:
                all_iterations = []
                best = None
                fin = open(a_file)
                for line in fin.readlines():
                    an_iteration = json.loads(line)
                    all_iterations.append(an_iteration)
                    if best == None or best["score"] > an_iteration["score"]:
                        best = an_iteration
                item["find_bump_parameters"].append({"all_iterations":all_iterations, "best":best})
            print("Loaded bumps from", item["dir"])

    def load_subs(self):
        for item in self.data:
            subs_name = os.path.join(item["dir"], "tmp/find_bump/subs.json")
            subs_str = open(subs_name).read()
            subs_json = json.loads(subs_str)
            item["subs"] = subs_json
            print("Loaded subs from", item["dir"])

    def do_plots(self):
        for plot_def in self.plot_definitions:
            try:
                self.do_one_axes(plot_def)
            except Exception:
                sys.excepthook(*sys.exc_info())
                print("Failed for plot definition\n", json.dumps(plot_def, indent=2))

    def transfer_matrix(self, tm, data_def):
        finder = QuadFinder(tm, data_def)
        #k = finder.solve_by_hand()
        k = finder.solve_analytically()
        return k

    def get_data(self, data_def):
        plot_data = []
        for item in self.data:
            if data_def["source"] == "subs":
                key = data_def["key"]
                plot_data.append(item["subs"][key])
            elif data_def["source"] == "parameters":
                key = data_def["key"]
                optimisation = data_def["optimisation_step"]-1
                if optimisation >= len(item["find_bump_parameters"]):
                    plot_data.append(0.0)
                    continue
                data_source = item["find_bump_parameters"][optimisation]["best"]["parameters"]
                for a_parameter in data_source:
                    if a_parameter["key"] == key:
                        plot_data.append(a_parameter["current_value"])
                        break
            elif data_def["source"] == "score":
                optimisation = data_def["optimisation_step"]-1
                optimisation = data_def["optimisation_step"]-1
                if optimisation >= len(item["find_bump_parameters"]):
                    plot_data.append(0.0)
                    continue
                data_source = item["find_bump_parameters"][optimisation]["best"]
                for score in data_source["score_list"]:
                    if score["score_type"] == data_def["score_type"] and \
                       score["station"] == data_def["station"]:
                        for key in data_def["key_list"]:
                            try:
                                score = score[key]
                            except KeyError:
                                print("key", key, "not in", score.keys(), "from", item["file_list"])
                                raise
                            except IndexError:
                                print("index", key, "not in item of length", len(score))
                                raise
                        if data_def["score_type"] == "tune":
                            score = hack_tune(score, data_def)
                        plot_data.append(score)
                        break
            elif data_def["source"] == "transfer_matrix":
                optimisation = data_def["optimisation_step"]-1
                data_source = item["find_bump_parameters"][optimisation]["best"]
                for score in data_source["score_list"]:
                    if score["score_type"] == "tune" and \
                       score["station"] == data_def["station"]:
                       plot_data.append(self.transfer_matrix(score["transfer_matrix"], data_def))
                       break
        if len(plot_data) != len(self.data):
            raise ValueError("Did not find plot_data for all items")
        return plot_data

    def get_cut(self, plot_def):
        will_cut_all = None
        for cut_def in plot_def["cut_list"]:
            cut_data = self.get_data(cut_def["cut_data"])
            will_cut_this = [cut_def["will_cut_lambda"](x) for x in cut_data]
            if will_cut_all == None:
                will_cut_all = will_cut_this
            else:
                will_cut_all = [will_cut_this[i] or will_cut_all[i] for i, x in enumerate(cut_data)]
        return will_cut_all

    def apply_cut(self, will_cut_data, x_data, y_data):
        if not will_cut_data:
            return x_data, y_data
        x_data = [x for i, x in enumerate(x_data) if will_cut_data[i]]
        y_data = [y for i, y in enumerate(y_data) if will_cut_data[i]]
        return x_data, y_data

    def setup_axes(self, axes, axis_def):
        axes.set_xlabel(axis_def["x_label"])
        axes.set_ylabel(axis_def["y_label"])
        if "title" in axis_def:
            axes.set_title(axis_def["title"])
        if "legend" in axis_def:
            axes.legend()
        if "verticals" in axis_def:
            lims = axes.get_ylim()
            for v in axis_def["verticals"]:
                axes.plot([v, v], lims, linestyle="--", c="lightgrey")
            axes.set_ylim(lims)
        if "horizontals" in axis_def:
            lims = axes.get_xlim()
            for v in axis_def["horizontals"]:
                axes.plot(lims, [v, v], linestyle="--", c="lightgrey")
            axes.set_xlim(lims)
        if "grid" in axis_def and axis_def["grid"]:
            axes.grid(True)
        utilities.setup_large_figure(axes)

    def do_one_axes(self, plot_def):
        figure = matplotlib.pyplot.figure(figsize=(20, 10))
        axes = figure.add_subplot(1, 1, 1)
        for dataset in plot_def["data"]:
            self.do_one_plot(axes, dataset, plot_def)
        self.setup_axes(axes, plot_def["axis"])
        figure.savefig(os.path.join(self.plot_dir, plot_def["file_name"]+".png"))

    def do_one_plot(self, axes, dataset, plot_def):
        x_data = self.get_data(dataset["x"])
        y_data = self.get_data(dataset["y"])
        will_cut_data = self.get_cut(plot_def)
        x_data, y_data = self.apply_cut(will_cut_data, x_data, y_data)
        x_data, y_data = self.sort_data(plot_def, x_data, y_data)
        label = None
        if "label" in dataset:
            label = dataset["label"]
            print(label, y_data)
        axes.plot(x_data, y_data, label=label)
        axes.scatter(x_data, y_data)


    def sort_data(self, plot_def, x_data, y_data):
        if "sort_axis" not in plot_def or not plot_def["sort_axis"]:
            return x_data, y_data
        if plot_def["sort_axis"] == "x":
            my_data = sorted(zip(x_data, y_data))
            #print(json.dumps(plot_def, indent=2))
            x_data, y_data = zip(*my_data)
        return x_data, y_data


def plot_definitions_ffync():
    cut_list = [{
            "cut_data":{"source":"score", "score_type":"tune", "station":1, "key_list":["weighted_tunes", 0], "optimisation_step":3},
            "will_cut_lambda":(lambda x: x < 1e-3),
    }]
    cut_list = []
    x_data = {"source":"subs", "key":"__energy__"}
    definitions = [{
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"score", "score_type":"tune", "station":1, "key_list":["tunes", 0], "optimisation_step":2},
                 "label":"$\\nu_x$",
               }, {"x":x_data,
                 "y":{"source":"score", "score_type":"tune", "station":1, "key_list":["tunes", 1], "optimisation_step":2},
                 "label":"$\\nu_y$"
               }],
        "file_name":"tune",
        "axis":{"x_label":"Energy [MeV]", "y_label":"$\\nu$", "title":"no quads", "legend":True}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"score", "score_type":"orbit", "station":1, "key_list":["orbit", 0], "optimisation_step":2},
               }],
        "file_name":"orbit_x",
        "axis":{"x_label":"Energy [MeV]", "y_label":"x [mm]"}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"score", "score_type":"orbit", "station":1, "key_list":["orbit", 1], "optimisation_step":2},
               }],
        "file_name":"orbit_xp",
        "axis":{"x_label":"Energy [MeV]", "y_label":"x'"}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"parameters", "optimisation_step":2, "key":"__h_bump_1_field__"},
               }],
        "file_name":"h_bump_1",
        "axis":{"x_label":"Energy [MeV]", "y_label":"D1 [T]"}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"parameters", "optimisation_step":2, "key":"__h_bump_delta_field__"},
               }],             
        "file_name":"h_bump_2",
        "axis":{"x_label":"Energy [MeV]", "y_label":"D2-D1 [T]"}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"parameters", "optimisation_step":2, "key":"__q1_strength__"},
               }],
        "file_name":"q1",
        "axis":{"x_label":"Energy [MeV]", "y_label":"Q1 [T/m]"}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"parameters", "optimisation_step":2, "key":"__q2_strength__"},
               }],
        "file_name":"q2",
        "axis":{"x_label":"Energy [MeV]", "y_label":"Q2 [T/m]"}
    }, {
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"transfer_matrix", "optimisation_step":2, "key":"q1", "station":1, "k1":0, "k2":0, "l_q":1.0, "l_d1":1.0, "l_d2":1.0, "nu_x":0.21, "nu_y":0.205},
                 "label":"f1"
               }, {"x":x_data,
                 "y":{"source":"transfer_matrix", "optimisation_step":2, "key":"q2", "station":1, "k1":0, "k2":0, "l_q":1.0, "l_d1":1.0, "l_d2":1.0, "nu_x":0.21, "nu_y":0.205},
                 "label":"f2"
               }],
        "file_name":"focal_length",
        "axis":{"x_label":"Energy [MeV]", "y_label":"Focal length [m]"}
    } ]
    return definitions


def plot_definitions_hffa(by):
    cut_list = []
    x_data = {"source":"score", "score_type":"orbit", "station":4, "key_list":["target", 0], "optimisation_step":1}
    definitions = [{
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"score", "score_type":"orbit", "station":4, "key_list":["orbit", 0], "optimisation_step":1},
                 "label":"$x$",
               },],
        "file_name":"position",
        "axis":{"x_label":"target r [mm]", "y_label":"found r [mm]", "grid":True, "title":f"b$_y$ {by} T"}
    },{
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"score", "score_type":"orbit", "station":4, "key_list":["orbit", 1], "optimisation_step":1},
                 "label":"$x'$",
               },],
        "file_name":"angle",
        "axis":{"x_label":"target r [mm]", "y_label":"found pr/pphi", "grid":True, "title":f"b$_y$ {by} T"}
    },{
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"parameters", "key":"__h_bump_1_field__", "optimisation_step":2},
                 "label":"$B1$",
               },{"x":x_data,
                 "y":{"source":"parameters", "key":"__h_bump_2_field__", "optimisation_step":2},
                 "label":"$B2$",
               },{"x":x_data,
                 "y":{"source":"parameters", "key":"__h_bump_3_field__", "optimisation_step":2},
                 "label":"$B3$",
               },{"x":x_data,
                 "y":{"source":"parameters", "key":"__h_bump_4_field__", "optimisation_step":2},
                 "label":"$B4$",
               },{"x":x_data,
                 "y":{"source":"parameters", "key":"__h_bump_5_field__", "optimisation_step":2},
                 "label":"$B5$",
               },],
        "file_name":"fields",
        "axis":{"x_label":"target r [mm]", "y_label":"Bump Field [T]", "title":f"b$_y$ {by} T"}
    },{
        "sort_axis":"x",
        "cut_list":cut_list,
        "data":[{"x":x_data,
                 "y":{"source":"score", "score_type":"tune", "station":8, "key_list":["tunes", 0], "optimisation_step":3},
                 "label":"$\\nu_x$",
               }, {"x":x_data,
                 "y":{"source":"score", "score_type":"tune", "station":8, "key_list":["tunes", 1], "optimisation_step":3},
                 "label":"$\\nu_y$"
               }],
        "file_name":"tune",
        "axis":{"x_label":"target r [mm]", "y_label":"$\\nu$", "title":f"b$_y$ {by} T", "legend":True, "horizontals":[0.0, +0.5, +1.0]}
    },]
    return definitions

def main_ffync():
    run_dir = "output/muon_ffynchrotron/bump_quest_v15/"
    plot_dir = os.path.join(run_dir, "plot_find_bump_parameters")
    file_glob = os.path.join(run_dir, "energy=*/find_bump_parameters*.out")
    plotter = PlotFindBumpParameters(plot_dir, file_glob)
    plotter.plot_definitions = plot_definitions_ffync()
    plotter.setup()
    plotter.do_plots()

def main_hffa():
    by = "0.20"
    k = "8.0095"
    run_dir = "output/2022-07-01_baseline/bump_quest_v10/"
    plot_dir = os.path.join(run_dir, "plot_find_bump_parameters_"+by+"_k="+k)
    file_glob = os.path.join(run_dir, "find_bump_r0*by="+by+"_k="+k+"/find_bump_parameters*.out")
    plotter = PlotFindBumpParameters(plot_dir, file_glob)
    plotter.plot_definitions = plot_definitions_hffa(by)
    plotter.setup()
    plotter.do_plots()

if __name__ == "__main__":
    main_hffa()
    matplotlib.pyplot.show(block=False)
    input("Done - Press <CR> to end")

