import ROOT
import xboa.common
import PyOpal.parser

root_objects = []

class GraphHacker():
    def __init__(self, color=1, width=1, style=1):
        self.color = color
        self.width = width
        self.style = style

    def hack(self, graph):
        graph.SetLineColor(self.color)
        graph.SetLineWidth(self.width)
        graph.SetLineStyle(self.style)

def load_file(file_name):
    fin = open(file_name)
    cols = range(10)
    var = [[] for a_col in cols]
    for number, line in enumerate(fin.readlines()):
        line = line.rstrip()
        line = line.split()
        for i, index in enumerate(cols):
            var[i].append(float(line[index]))
        if abs(float(line[0])-3.0) < 0.1:
            print("near 3:", float(line[0]))
    return var

def do_fit(delta, graph):
    eqn = "0.5*[0]*(tanh((x-0.5*[1]-[3])/[2])-tanh((x+0.5*[1]-[3])/[2]))"
    name = "fa1"+str(len(root_objects))
    print("Fitting with delta", delta)
    fit = ROOT.TF1(name, eqn, delta-1.0, delta+1.0)
    fit.SetParameter(0, 1)
    fit.SetParameter(1, 0.6)
    fit.SetParameter(2, 0.3)
    fit.SetParameter(3, delta)
    fit.SetLineColor(ROOT.kGreen)
    fit.SetLineStyle(3)
    graph.Fit(fit)
    fit.Draw("SAME")
    root_objects.append(fit)
    print()

def load_opal_field(file_name):
    PyOpal.parser.initialise_from_opal_file("VerticalFFAGMagnet.tmp")

def plot_field(field, axis, canvas, scale, offset, graph_hacker):
    f_list = [(x-offset)*scale for x in field[axis]]
    name = "(machida col "+str(axis)
    if offset != 0.:
        name += " -"+str(offset)
    name += ")"
    if scale != 1.:
        name += "x"+str(scale)
    print("axis", axis, ":", field[axis][0])
    hist, graph = xboa.common.make_root_graph(name, field[0], "s [m]",
                                              f_list, "field [T]")
    if canvas == None:
        canvas = ROOT.TCanvas("field", "field", 1000, 1000)
        canvas.Draw()
        root_objects.append(canvas)
        hist.Draw()
    canvas.cd()
    graph_hacker.hack(graph)
    graph.Draw("SAME L")
    return canvas

def horizontal_lines():
    y_val = 0.347745*10
    z0, z1 = -1., 10.
    for y in [y_val, -y_val]:
        graph = ROOT.TGraph(2)
        graph.SetPoint(0, z0, y)
        graph.SetPoint(1, z1, y)
        graph.SetLineColor(ROOT.kGray)
        graph.Draw("L SAME")
        root_objects.append(graph)

def main():
    field = load_file("/home/vol08/scarf148/data/work/2017-07-07_isis2/machida_field/machida_field.dat")
    canvas = None
    canvas = plot_field(field, 7, canvas, 1., 0., GraphHacker(ROOT.kRed, 1, 2))
    canvas = plot_field(field, 8, canvas, 1., 0., GraphHacker(ROOT.kMagenta, 1, 2))
    canvas = plot_field(field, 9, canvas, 1., 0., GraphHacker(ROOT.kBlue, 1, 2))
    canvas = plot_field(field, 2, canvas, 10, 0., GraphHacker(ROOT.kCyan, 1, 2))
    canvas = plot_field(field, 4, canvas, 100, 0.62489, GraphHacker(ROOT.kOrange, 1, 2))
    horizontal_lines()
    return canvas

if __name__ == "__main__":
    main()