import ROOT
import xboa.common

class PhaseSpacePlots(object):
    def __init__(self, config):
        self.config = config
        self.single_turn_plot_list = [-1]+self.config.track_beam["single_turn_plots"]
        self.x_graphs = [ROOT.TGraph() for i in self.single_turn_plot_list]
        self.y_graphs = [ROOT.TGraph() for i in self.single_turn_plot_list]
        self.z_graphs = [ROOT.TGraph() for i in self.single_turn_plot_list]
        self.x_hist = None
        self.y_hist = None
        self.z_hist = None
        self.x_canvas = None
        self.y_canvas = None
        self.z_canvas = None
        co_seed = config.find_closed_orbits["seed"]
        self.turn_index = {}
        self.min_r = self.config.track_beam["min_radius"]
        self.max_p = self.config.track_beam["max_delta_p"]


    def process_hit(self, event, hit):
        if hit['x'] < self.min_r or abs(hit['py']) > self.max_p:
            return
        if event not in self.turn_index:
              self.turn_index[event] = 0
        turn_index = self.turn_index[event]
        x, px, y, py, ct, pz = hit['x'], hit['px'], hit['y'], hit['py'], hit['ct'], hit['pz']
        if turn_index in self.single_turn_plot_list:
            index = self.single_turn_plot_list.index(turn_index)
            self.x_graphs[index].SetPoint(self.x_graphs[index].GetN(), hit['x'], hit['px'])
            self.y_graphs[index].SetPoint(self.y_graphs[index].GetN(), hit['y'], hit['py'])
            self.z_graphs[index].SetPoint(self.z_graphs[index].GetN(), hit['ct'], hit['pz'])
        self.x_graphs[0].SetPoint(self.x_graphs[0].GetN(), hit['x'], hit['px'])
        self.y_graphs[0].SetPoint(self.y_graphs[0].GetN(), hit['y'], hit['py'])
        self.z_graphs[0].SetPoint(self.z_graphs[0].GetN(), hit['ct'], hit['pz'])
        self.turn_index[event] += 1

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
      
    def finalise(self):
        output_dir = self.config.run_control["output_dir"]

        self.x_canvas = xboa.common.make_root_canvas("x phase space")
        self.x_canvas.Draw()
        for index, graph in enumerate(reversed(self.x_graphs)):
            i = len(self.x_graphs)-index
            if graph.GetN() == 0:
                continue
            self.x_hist = self.axes("x", graph)
            self.x_hist.Draw()
            self.set_marker(graph)
            graph.Draw("psame")
            self.x_canvas.Update()
            for format in "png", "eps", "root":
                name = output_dir+"/x-px_phase-space_"+str(i)+"."+format
                self.x_canvas.Print(name)

        self.y_canvas = xboa.common.make_root_canvas("y phase space")
        self.y_canvas.Draw()

        for index, graph in enumerate(reversed(self.y_graphs)):
            i = len(self.y_graphs)-index
            if graph.GetN() == 0:
                continue
            self.y_hist = self.axes("y", graph)
            self.y_hist.Draw()
            self.set_marker(graph)
            graph.Draw("psame")
            self.y_canvas.Update()
            for format in "png", "eps", "root":
                name = output_dir+"/y-py_phase-space_"+str(i)+"."+format
                self.y_canvas.Print(name)

        self.z_canvas = xboa.common.make_root_canvas("z phase space")
        self.z_canvas.Draw()

        for index, graph in enumerate(reversed(self.z_graphs)):
            i = len(self.z_graphs)-index
            if graph.GetN() == 0:
                continue
            self.z_hist = self.axes("z", graph)
            self.z_hist.Draw()
            self.set_marker(graph)
            graph.Draw("psame")
            self.z_canvas.Update()
            for format in "png", "eps", "root":
                name = output_dir+"/ct-pz_phase-space_"+str(i)+"."+format
                self.z_canvas.Print(name)
