#This file is a part of xboa
#
#xboa is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#xboa is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with xboa in the doc folder.  If not, see 
#<http://www.gnu.org/licenses/>.

"""
\namespace _opal_tracking
"""

import math
import numpy
import ROOT

import xboa.common

class TunesAnalysis(object):
    def __init__(self, config):
        self.config = config
        self.beam_centre = None
        self.beam_ellipse = None
        self.coordinates = ['x', 'px', 'y', 'py', 'ct', 'pz']
        self.real_points = {}
        self.cholesky_points = {}
        self.x_tunes = {}
        self.x_actions = {}
        self.y_tunes = {}
        self.y_actions = {}
        self.fourd_actions = {}
        self.plot_events = self.config.track_beam["plot_events"]
        self.real_graph_list = [ROOT.TGraph() for i in self.plot_events]
        self.chol_graph_list = [ROOT.TGraph() for i in self.plot_events]

    def process_hit(self, event, hit):
        if not event in self.real_points:
            self.real_points[event] = []
            self.cholesky_points[event] = []
            self.x_tunes[event] = []
            self.y_tunes[event] = []
            self.x_actions[event] = []
            self.y_actions[event] = []
        hit = [hit[var] for var in self.coordinates]
        if hit[0] < self.config.track_beam["min_radius"] or \
           abs(hit[1]) > self.config.track_beam["max_delta_p"]:
            return
        self.real_points[event].append(hit)
        self.real_points[event] = sorted(self.real_points[event], key = lambda hit: hit[4])
        chol_hit = self._cholesky_conversion(hit)
        self.cholesky_points[event].append(chol_hit)
        self.cholesky_points[event] = sorted(self.cholesky_points[event], key = lambda hit: hit[4])
        if event in self.plot_events:
           graph_id = self.plot_events.index(event)
           real_graph = self.real_graph_list[graph_id]
           real_graph.SetPoint(real_graph.GetN(), hit[0], hit[1])
           chol_graph = self.chol_graph_list[graph_id]
           chol_graph.SetPoint(chol_graph.GetN(), chol_hit[0], chol_hit[1])
        #print "...", hit[2], hit[3]
        self.x_actions[event].append(self.get_action(self.cholesky_points[event][-1], 0))
        self.y_actions[event].append(self.get_action(self.cholesky_points[event][-1], 2))
        if len(self.real_points[event]) > 2:
            self.get_tunes(event, self.cholesky_points[event][-2], self.cholesky_points[event][-1])

    def get_action(self, hit, ax):
        action = numpy.dot(numpy.transpose(hit[ax:ax+2]), self.cholesky_inv[ax:ax+2, ax:ax+2])
        action = numpy.dot(action, hit[ax:ax+2])
        return action

    def get_tunes(self, event, hit0, hit1):
        delta = hit1 - hit0
        phi0_x = math.atan2(hit0[1], hit0[0])
        phi1_x = math.atan2(hit1[1], hit1[0])
        dphi_x = (phi1_x-phi0_x)/math.pi/2.
        while dphi_x < 0:
            dphi_x += 1.
        phi0_y = math.atan2(hit0[3], hit0[2])
        phi1_y = math.atan2(hit1[3], hit1[2])
        dphi_y = (phi1_y-phi0_y)/math.pi/2.
        while dphi_y < 0:
            dphi_y += 1.
        self.x_tunes[event].append(1.-dphi_x)
        self.y_tunes[event].append(1.-dphi_y)

    def _cholesky_conversion(self, hit):
        hit_array = numpy.array(hit) #hit[var] for var in self.coordinates])
        hit_array[:4] -= self.beam_centre
        hit_array[:4] = numpy.dot(self.cholesky_inv, hit_array[:4])
        return hit_array

    def set_match(self, mean, ellipse):
        self.beam_ellipse = ellipse
        self.beam_centre = mean
        cholesky = numpy.linalg.cholesky(self.beam_ellipse)
        cholesky /= numpy.linalg.det(cholesky)**(1./len(self.coordinates))
        self.cholesky_inv = numpy.linalg.inv(cholesky)
        self.cholesky_x = numpy.linalg.inv(cholesky[0:2, 0:2])
        self.cholesky_y = numpy.linalg.inv(cholesky[2:4, 2:4])
        print("Setup cholesky inverse to")
        print(self.cholesky_inv)

    def graph_range(self, axis):
        x_min = axis.GetXmin()
        x_max = axis.GetXmax()
        x_min -= (x_max-x_min)*0.1
        x_max += (x_max-x_min)*0.1
        return x_min, x_max

    def axes(self, axis, graph):
        if axis == "x":
            label = ";x [mm];p_{x} [MeV/c]"
        elif axis == "y":
            label = ";y [mm];p_{y} [MeV/c]"
        else:
            label = ";ct [mm];p_{z} [MeV/c]"
        x_min, x_max = self.graph_range(graph.GetXaxis())
        y_min, y_max = self.graph_range(graph.GetYaxis())
        x_hist = ROOT.TH2D("", label,
                           1000, x_min, x_max,
                           1000, y_min, y_max)
        x_hist.SetStats(False)
        return x_hist

    def set_marker(self, graph):
        if graph.GetN() < 100:
            graph.SetMarkerStyle(24)
        elif graph.GetN() < 1000:
            graph.SetMarkerStyle(7)
        elif graph.GetN() < 10000:
            graph.SetMarkerStyle(6)
        else:
            graph.SetMarkerStyle(1)

    def draw_plots(self):
        output_dir = self.config.run_control["output_dir"]

        for prefix, graph_list in ("real", self.real_graph_list), ("chol", self.chol_graph_list):
            self.x_canvas = xboa.common.make_root_canvas("x phase space")
            self.x_canvas.Draw()
            for index, graph in enumerate(reversed(graph_list)):
                i = len(graph_list)-index
                if graph.GetN() == 0:
                    continue
                self.x_hist = self.axes("x", graph)
                self.x_hist.Draw()
                self.set_marker(graph)
                graph.Draw("psame")
                self.x_canvas.Update()
                for format in "png", "eps", "root":
                    name = output_dir+"/"+prefix+"_x-px_phase-space_"+str(i)+"."+format
                    self.x_canvas.Print(name)

    def necktie_graph(self):
        output_dir = self.config.run_control["output_dir"]
        x_tunes, y_tunes = [], []
        for event in list(self.real_points.keys()):
            if len(self.x_tunes[event]) > 0 and len(self.y_tunes[event]) > 0:
                #if event in self.plot_events:
                print("mean x:", numpy.mean(self.x_tunes[event]), "std x:", numpy.std(self.x_tunes[event]), "mean y:", numpy.mean(self.y_tunes[event]), "std y:", numpy.std(self.y_tunes[event]))
                x_tunes.append(numpy.mean(self.x_tunes[event]))
                y_tunes.append(numpy.mean(self.y_tunes[event]))
        canvas = xboa.common.make_root_canvas("necktie")
        hist, graph = xboa.common.make_root_graph("", x_tunes, "", y_tunes, "")
        hist.Draw()
        graph.Draw("SAMEP")
        for format in "png", "eps", "root":
            canvas.Print(output_dir+"/necktie_graph."+format)
        canvas = xboa.common.make_root_canvas("necktie")
        hist = xboa.common.make_root_histogram("", x_tunes, "", 100, y_tunes, "", 100, xmin=0.5, xmax=0.7, ymin=0.6, ymax=0.65)
        hist.Draw("COLZ")
        for format in "png", "eps", "root":
            canvas.Print(output_dir+"/necktie_hist."+format)

    def finalise(self):
        for event in sorted(self.real_points.keys()):
            if event not in self.plot_events:
               continue
            print(event)
            for i, a_point in enumerate(self.real_points[event]):
                for x in a_point:
                    print(str(round(x, 2)).rjust(8), end=' ')
                for x in self.cholesky_points[event][i]:
                    print(str(round(x, 2)).rjust(8), end=' ')
                if i > 2:
                    print(self.x_tunes[event][i-2], self.y_tunes[event][i-2])
                else:
                    print()
        self.draw_plots()
        self.necktie_graph()
        return

    def plot_tunes(self):
        pass
