import glob
import os
import json
import sys
import math
import numpy
import argparse

import ROOT
import xboa.common
import utils.utilities as utilities
from xboa.hit import Hit
from utils.decoupled_transfer_matrix import DecoupledTransferMatrix


class ClosedOrbitsPlotter(object):
    def __init__(self):
        self.data = []
        self.tm_lambda = None
        self.phase_advance_axis = 0
        self.log_x = False
        self.subs_key = "substitutions"

    def load_file(self, file_name):
        print("Loading", file_name)
        fin = open(file_name)
        my_data = []
        my_data = [json.loads(line) for line in fin.readlines()]
        flattened = []
        for item in my_data:
            flattened += item
            user_def = file_name.split("_x_")[1]
            user_def = user_def.split("_y_")[0]
            user_def = float(user_def)
            for thing in item:
                thing["substitutions"]["__position [mm]__"] = user_def
        self.tm_lambda = lambda tm: print("Rotation R:\n", numpy.real(tm.r),
                                          "\nM Symplecticity:\n", tm.symplecticity(tm.m),
                                          "\nM Det:", numpy.linalg.det(tm.m),
                                          "\nR Det:", numpy.linalg.det(tm.r))
        #self.get_tm(flattened[0])
        self.data += flattened
        print("Types", [type(dat) for dat in self.data])

    def do_plots_all_in(self):
        self.plot_item_2("x", lambda item: item["seed"][0],
                         "y", lambda item: item["seed"][2],
                         "delta", self.get_co, None)
        self.plot_item_2("x", lambda item: item["seed"][0],
                         "y", lambda item: item["seed"][2],
                         "delta", self.get_co, None)

    def do_plots_by_sub_2(self):
        self.tm_lambda = self.tm_det
        name = "abs(1-Determinant)"
        canvas, mgraph = self.plot_item(name, self.get_tm, axis, None, None)
        mgraph.Draw("AP")
        canvas.SetLogx()
        canvas.SetLogy()
        canvas.Print(self.output_dir+"/"+axis_name+"_vs_determinant.png")

        self.tm_lambda = self.tm_sym_xxp
        name = "J(0,1)"
        canvas, mgraph = self.plot_item(name, self.get_tm, axis, None, None)
        mgraph.Draw("AP")
        canvas.Print(self.output_dir+"/"+axis_name+"_vs_determinant.png")

    def do_plots_by_sub(self):
        subs_axes = utilities.get_substitutions_axis(self.data, self.subs_key)
        print(subs_axes)
        for axis in subs_axes:
            axis_name = utilities.sub_to_name(axis).lower().replace(" ", "_")
            canvas, mgraph = None, None
            for self.phase_advance_axis in 0, 1:
                self.tm_lambda = self.tm_phase_advance
                name = "Phase Advance "+str(self.phase_advance_axis)
                canvas, mgraph = self.plot_item(name, self.get_tm, axis, canvas, mgraph)
            mgraph.Draw("AP")
            canvas.Print(self.output_dir+"/"+axis_name+"_vs_phase_advance.png")

            continue
            for name in "x", "x'", "y", "y'":
                canvas, mgraph = self.plot_item(
                    name,
                    lambda item: Hit.new_from_dict(item["seed_hit"])[name],
                    axis, None, None)
                mgraph.Draw("AP")
                name = name.replace("'", "p")
                canvas.Print(self.output_dir+"/"+axis_name+"_vs_"+name+".png")

            
            for self.phase_advance_axis in 0, 1:
                canvas, mgraph = None, None
                self.tm_lambda = self.tm_phase_advance
                name = "phase advance "+str(self.phase_advance_axis)
                canvas, mgraph = self.plot_item(name, self.get_tm, axis, canvas, mgraph)
                mgraph.Draw("AP")
                name = name.replace(" ", "_")
                canvas.Print(self.output_dir+"/"+axis_name+"_vs_"+name+".png")


            name = "delta"
            canvas, mgraph = self.plot_item(name, self.get_co, axis, None, None)
            mgraph.Draw("AP")
            canvas.SetLogx()
            canvas.SetLogy()
            canvas.Print(self.output_dir+"/"+axis_name+"_vs_delta.png")

            name = "abs(J(0, 1))"
            self.tm_lambda = self.tm_symplex_01
            canvas, mgraph = self.plot_item(name, self.get_tm, axis, None, None)
            mgraph.Draw("AP")
            canvas.SetLogx()
            canvas.Print(self.output_dir+"/j01_vs_delta.png")


            for var in ["x", "x'", "y", "y'"]:
                self.sigma_var = var
                name = "sigma_"+var
                canvas, mgraph = self.plot_item(name, self.get_sigma, axis, None, None)
                mgraph.Draw("AP")
                canvas.SetLogx()
                canvas.SetLogy()
                canvas.Print(self.output_dir+"/"+axis_name+"_vs_"+name+".png")




    def plot_item_2(self, x_name, x_lambda, y_name, y_lambda, z_name, z_lambda, canvas):
        name = x_name+" vs "+y_name+" vs "+z_name
        x_data, y_data, z_data = [], [], []
        for item in self.data:
            x_i, y_i, z_i = 0., 0., 0.
            try:
                x_i = x_lambda(item)
                y_i = y_lambda(item)
                z_i = z_lambda(item)
            except Exception:
                sys.excepthook(*sys.exc_info())
            x_data.append(x_i)
            y_data.append(y_i)
            z_data.append(z_i)
        print("Plotting ", name)
        print("   ", x_name, x_data)
        print("   ", y_name, y_data)
        print("   ", z_name, z_data)

        n_points = len(self.data)
        if canvas == None:
            canvas = xboa.common.make_root_canvas(name)
            canvas.Draw()
            x_min_max = xboa.common.min_max(x_data)
            y_min_max = xboa.common.min_max(y_data)
            z_min_max = xboa.common.min_max(z_data)
            hist = ROOT.TH3D("", "",
                             10, x_min_max[0], x_min_max[1],
                             10, y_min_max[0], y_min_max[1],
                             10, z_min_max[0], z_min_max[1],
                    )
            hist.GetXaxis().SetTitleOffset(3)
            hist.SetStats(False)
            hist.Draw()
            self.root_objects.append(hist)
        canvas.cd()
        graph2d = ROOT.TGraph2D(n_points)
        self.root_objects.append(graph2d)
        #ROOT.gStyle.SetPalette(1)
        graph2d.SetMarkerStyle(20)
        graph2d.SetTitle("")
        graph2d.Draw('same pcolz')
        for i in range(n_points):
            graph2d.SetPoint(i, x_data[i], y_data[i], z_data[i])
        name = name.replace(" ", "_")
        canvas.SetTheta(90)
        canvas.SetPhi(-90)
        canvas.SetLogz()
        canvas.Print(self.output_dir+"/"+name+".png")
        return canvas, graph2d

    def plot_item(self, y_name, get_item_lambda, axis, canvas, mgraph):
        x_data = [item[self.subs_key][axis] for item in self.data]
        x_name = utilities.sub_to_name(axis)+utilities.sub_to_units(axis)
        y_data = []
        for item in self.data:
            try:
                y_data.append(get_item_lambda(item))
            except Exception:
                sys.excepthook(*sys.exc_info())
                y_data.append(0.)
        name = utilities.sub_to_name(axis)+" vs "+y_name
        print("   ", y_name, y_data)
        hist, graph = xboa.common.make_root_graph(name, x_data, x_name, y_data, y_name)
        if canvas == None:
            canvas = xboa.common.make_root_canvas(name)
            if self.log_x:
                canvas.SetLogx(True)
            canvas.Draw()
            mgraph = ROOT.TMultiGraph()
            mgraph.SetTitle(";"+x_name+";"+y_name)
            self.root_objects.append(mgraph)
        if not mgraph.GetListOfGraphs():
            index = 0
        else:
            index = mgraph.GetListOfGraphs().GetEntries()
        color_list = [ROOT.kRed, ROOT.kBlue]
        if len(x_data) < 25:
            style_list = [20, 20]
        else:
            style_list = [7, 7]
        graph.SetMarkerColor(color_list[index])
        graph.SetMarkerStyle(style_list[index])
        mgraph.Add(graph)
        return canvas, mgraph

    def get_tm(self, item):
        tm = item["tm"]
        tm = [row[1:5] for row in tm]
        try:
            tm = DecoupledTransferMatrix(tm)
        except Exception:
            sys.excepthook(*sys.exc_info())
            return 0.
        return self.tm_lambda(tm)

    def tm_det(self, tm):
        det =  abs(1-numpy.linalg.det(tm.m))
        return det

    def tm_phase_advance(self, tm):
        pa = [tm.get_phase_advance(j) for j in range(2)]
        pa = sorted(pa)
        pa = abs(pa[self.phase_advance_axis]/2./math.pi)
        return pa

    def tm_symplex_01(self, tm):
        print("Symplecticity\n", tm.symplecticity(tm.m))
        return abs(tm.symplecticity(tm.m)[0][1])

    def get_co(self, item):
        var_list = ["x", "x'", "y", "y'"]
        units = {"x":1., "x'":1e3, "y":1., "y'":1e3}
        print("Ref track length", len(item["ref_track"]), item["ref_track"][0])
        print(item.keys())
        ref_0 = Hit.new_from_dict(item["ref_track"][0])
        ref_1 = Hit.new_from_dict(item["ref_track"][2])
        delta = [(ref_0[var]-ref_1[var])*units[var] for var in var_list]
        delta_mag = sum([x*x for x in delta])**0.5
        return delta_mag

    def get_sigma(self, item):
        ref = [Hit.new_from_dict(hit_dict)[self.sigma_var] for hit_dict in item["ref_track"]]
        return numpy.std(ref)

    def parse_args(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('file_names', metavar='N', type=str, nargs='+')
        parser.add_argument('--log_x', dest='log_x', action='store_true')
        parser.set_defaults(log_x=False)
        args = parser.parse_args()
        self.output_dir = os.path.split(args.file_names[0])[0]
        self.output_dir = os.path.join(self.output_dir, "plot_closed_orbits")
        utilities.clear_dir(self.output_dir)
        for file_name in args.file_names:
            self.load_file(file_name)
        self.log_x = args.log_x


    root_objects = []

def main():
    DecoupledTransferMatrix.det_tolerance = 1e6
    utilities.setup_gstyle()
    if len(sys.argv[1:]) == 0:
        print("Usage - python plot_closed_orbits [file_name_1] [file_name_2] ...")
    co_plotter = ClosedOrbitsPlotter()
    co_plotter.parse_args(sys.argv)
    co_plotter.do_plots_by_sub()
    #co_plotter.do_plots_all_in()

if __name__ == "__main__":
    main()
    input("Done")
