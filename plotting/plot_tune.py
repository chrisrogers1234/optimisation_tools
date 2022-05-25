import shutil
import os
import glob
import sys
import json
import math
import xboa.common
import numpy
import ROOT

from utils import utilities

ROOT_OBJECTS = []

def load_file(file_name):
    fin = open(file_name)
    data = [json.loads(line) for line in fin.readlines()]
    return data

def clean_tune_data(data, verbose=False):
    old_mean = 0
    old_std = 0
    keep_going = True
    my_data = [(0, item) for item in data if abs(item) > 1e-19]
    while keep_going:
        data_tmp = [value for delta, value in my_data]
        mean = numpy.mean(data_tmp)
        std = numpy.std(data_tmp)
        if verbose:
            print(len(my_data), mean, std, abs(std - old_std)/std, abs(mean - old_mean)/std)
        if verbose:
            print(keep_going)
        old_std = std
        old_mean = mean
        my_data = [(abs(value - mean), value) for delta, value in my_data]
        my_data = sorted(my_data)
        if verbose:
            print("   ", my_data[-3:])
        keep_going = len(my_data) > 2 and my_data[-1][0] > 5*std
        my_data = my_data[:-1]
    my_data = [item[1] for item in my_data]
    if verbose:
        print("DATA IN",  sorted(data))
        print("DATA OUT", my_data)
        print("MEAN", numpy.mean(my_data))
    return my_data

def get_axes(axis_key, group_dict, tune_axis):
    x_name = utilities.sub_to_name(axis_key)
    canvas_name = tune_axis+' vs '+x_name
    canvas = xboa.common.make_root_canvas(canvas_name)
    tune = []
    ordinate = []
    if x_name in units:
        x_name += " ["+units[x_name]+"]"
    for group_key in group_dict:
        tune += group_dict[group_key]['x_tune']+group_dict[group_key]['y_tune']
        ordinate += group_dict[group_key]['axes'][axis_key]
    y_min, y_max = min(tune), max(tune)
    x_min, x_max = min(ordinate), max(ordinate)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    dx, dy = x_max-x_min, y_max-y_min
    x_min -= dx*0.1
    x_max += dx*0.4
    y_min -= dy*0.1
    y_max += dy*0.1
    hist = xboa.common.make_root_histogram('axes', 
                                           [x_min-1], x_name, 1000,
                                           [y_min-1], tune_axis, 1000,
                                           xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)
    hist.Draw()
    return canvas, hist

def make_root_graph_errors(name, x_values, y_values, y_errors):
    graph = ROOT.TGraphErrors()
    ROOT_OBJECTS.append(graph)
    n_points = len(x_values)
    sorted_values = [None]*n_points
    for i in range(n_points):
        sorted_values[i] = (x_values[i], y_values[i], y_errors[i])
    sorted_values = sorted(sorted_values)
    for i in range(n_points):
        graph.SetPoint(i, sorted_values[i][0], sorted_values[i][1])
        graph.SetPointError(i, 0., 0.)#sorted_values[i][2])
        print(sorted_values[i])
    graph.SetName(name)
    return graph

def do_one_1d_plot(axis_key, group_dict, plot_dir, tune_axis):
    canvas, hist = get_axes(axis_key, group_dict, tune_axis)
    leg_list = []
    y_color = numpy.array([ROOT.kBlue, ROOT.kCyan+2, ROOT.kBlue-7, ROOT.kGreen+1, ROOT.kGreen+3])
    x_color = numpy.array([ROOT.kRed, ROOT.kRed+3, ROOT.kYellow-3, ROOT.kMagenta+1, ROOT.kMagenta+3])
    for group_key in sorted(group_dict.keys()):
        item = group_dict[group_key]
        graph_x = make_root_graph_errors(group_key+" #nu_{x}", item['axes'][axis_key], item['x_tune'], item['x_tune_err'])
        graph_y = make_root_graph_errors(group_key+" #nu_{y}", item['axes'][axis_key], item['y_tune'], item['y_tune_err'])
        print("For group", group_key, "found", len(item['axes'][axis_key]), "items")
        print("   ", axis_key, item['axes'][axis_key])
        print("    x tune", item['x_tune'])
        print("    y tune", item['y_tune'])
        graph_x.SetMarkerStyle(24)
        graph_x.SetMarkerColor(x_color[0])
        graph_y.SetMarkerStyle(26)
        graph_y.SetMarkerColor(y_color[0])
        graph_x.Draw("P SAME")
        graph_y.Draw("P SAME")
        leg_list += [graph_x, graph_y]
        numpy.roll(x_color, 1)
        numpy.roll(y_color, 1)
    legend = xboa.common.make_root_legend(canvas, leg_list)
    legend.Draw()
    legend.SetBorderSize(1)
    legend.SetX1NDC(0.75)
    legend.SetX2NDC(0.9)
    legend.SetY2NDC(0.9)
    legend.SetY1NDC(0.5)
    canvas.Update()
    canvas_name = canvas.GetTitle().replace(" ", "_")
    for format in "eps", "png", "root":
        canvas.Print(plot_dir+"/"+canvas_name+"."+format)

def do_ellipse_plot(axis_key, group_dict, plot_dir, tune_axis):
    leg_list = []
    color = numpy.array([ROOT.kBlue+3, ROOT.kCyan+2, ROOT.kBlue-7, ROOT.kGreen+1, ROOT.kGreen+3]+
                        [ROOT.kRed+3, ROOT.kRed, ROOT.kYellow-3, ROOT.kMagenta+1, ROOT.kMagenta+3])
    multigraph = ROOT.TMultiGraph()
    for group_key in sorted(group_dict.keys()):
        for item in group_dict[group_key]['y_signal']:
            #print item
            x_list = [u[0] for u in item]
            y_list = [u[1] for u in item]
            hist, graph = xboa.common.make_root_graph(group_key, x_list, '[mm]', y_list, '[MeV/c]', sort=False)
            graph.SetMarkerStyle(24)
            graph.SetMarkerColor(color[0])
            multigraph.Add(graph)
            numpy.roll(color, 1)
        multigraph.Draw()
    canvas = xboa.common.make_root_canvas("phase space")
    canvas.Update()
    canvas_name = canvas.GetTitle().replace(" ", "_")
    for format in "eps", "png", "root":
        canvas.Print(plot_dir+"/"+canvas_name+"."+format)

def get_groups(data, group_axis):
    axis_candidates = utilities.get_substitutions_axis(data, "substitutions")
    if group_axis != None and group_axis not in axis_candidates:
        raise RuntimeError("Did not recognise group axis "+str(group_axis))
    if group_axis == None:
        group_list = [item for item in data]
        return {"":{"item_list":[i for i in range(len(data))]}}
    else:
        # group_list is lists the possible groups
        # e.g. list of all possible values of "__bump_field_1__"
        group_list = [item['substitutions'][group_axis] for item in data]
        group_list = list(set(group_list)) # unique list
        # tmp_group_dict is mapping from group value to the items having that value
        tmp_group_dict = dict([(group, []) for group in group_list])
        for key in tmp_group_dict:
            tmp_group_dict[key] = [i for i, item in enumerate(data) \
                                        if item['substitutions'][group_axis] == key]
    group_dict = {}
    for key in tmp_group_dict:
        new_key = utilities.sub_to_name(group_axis)+" "+format(key, "3.3g")
        group_dict[new_key] = {'item_list':tmp_group_dict[key]}
        print(new_key, ":", group_dict[new_key])
    return group_dict

def plot_data_1d(data, tune_axis, plot_dir, group_axis = None, cell_conversion = 1, skip_length_one = True):
    """
    Plot x and y tune against a given parameter, grouping by "group_axis"
      - data: the tune data
      - tune_axis: string ... ring tune or cell tune
      - plot_dir: directory to place the plot
      - group_axis: set to None to ignore; set to a substitution variable to
                    place tune items in the same graph if the group_axis is the
                    same
      - cell_conversion: scale the tunes by the "cell_conversion" factor
      - skip_length_one: ignore a graph if there is only one element
    """
    axis_candidates = utilities.get_substitutions_axis(data, "substitutions")
    if group_axis in axis_candidates:
        del axis_candidates[group_axis]
    group_dict = get_groups(data, group_axis)
    group_key_list = list(group_dict.keys())
    for group_key in group_key_list:
        my_axes = dict([(key, []) for key in axis_candidates])
        x_signal, y_signal = [], []
        x_tune, y_tune = [], []
        x_tune_err, y_tune_err = [], []
        item_list = group_dict[group_key]['item_list']
        for i in item_list:
            item = data[i]
            verbose = False
            x_dphi = clean_tune_data(item['u_dphi'], verbose)
            y_dphi = clean_tune_data(item['v_dphi'], verbose)
            if verbose:
                print("X DPHI", sorted(x_dphi))
                print("Y DPHI", sorted(y_dphi))
            x_tune.append(numpy.mean(x_dphi))
            y_tune.append(numpy.mean(y_dphi))
            x_signal.append(item['u_signal'])
            y_signal.append(item['v_signal'])

            if len(x_dphi) > 1:
                x_tune_err.append(numpy.std(x_dphi))
            else:
                x_tune_err.append(1.)
            if len(y_dphi) > 1:
                y_tune_err.append(numpy.std(y_dphi))
            else:
                y_tune_err.append(1.)

            for key in my_axes:
                my_axes[key].append(item['substitutions'][key])
            if cell_conversion != 1:
                x_tune = [nu*cell_conversion - math.floor(nu*cell_conversion) for nu in x_tune]
                y_tune = [nu*cell_conversion - math.floor(nu*cell_conversion) for nu in y_tune]
        if len(x_tune) == 1 and skip_length_one:
            del group_dict[group_key]
            print("Removed", group_key, "because length was 1")
            continue

        group_dict[group_key]['x_tune'] = x_tune
        group_dict[group_key]['x_tune_err'] = x_tune_err
        group_dict[group_key]['y_tune'] = y_tune
        group_dict[group_key]['y_tune_err'] = y_tune_err
        group_dict[group_key]['axes'] = my_axes
        group_dict[group_key]['x_signal'] = x_signal
        group_dict[group_key]['y_signal'] = y_signal

    for key in axis_candidates:
        do_one_1d_plot(key, group_dict, plot_dir, tune_axis)
        do_ellipse_plot(key, group_dict, plot_dir, tune_axis)
    return

def plot_tune_network(self, canvas):
    for x_index in range(1, 4):
        for y_index in range(1, 4):
            x_values = [1./x_index, 1./x_index]
            y_values = [0, 1.]

def plot_data_2d(data, tune_axis, plot_dir):
    x_tune = [item['u_tune'] for item in data]
    y_tune = [item['v_tune'] for item in data]
    print("X Tunes", x_tune)
    print("Y Tunes", y_tune)
    x_name = "Varying "
    axis_candidates = utilities.get_substitutions_axis(data, "substitutions")
    for key in axis_candidates:
        min_val = str(min(axis_candidates[key]))
        max_val = str(max(axis_candidates[key]))
        x_name += utilities.sub_to_name(key)+" from "+min_val+" to "+max_val
    axis_candidates = utilities.get_substitutions_axis(data, "substitutions")
    canvas = xboa.common.make_root_canvas('tune x vs y')
    hist, graph = xboa.common.make_root_graph('tune x vs y', x_tune, "horizontal tune", y_tune, "vertical tune", xmin=0., xmax=1., ymin=0., ymax=1.)
    hist.SetTitle(x_name)
    hist.Draw()
    #utilities.tune_lines(canvas)
    graph.SetMarkerStyle(24)
    graph.Draw("P SAME")
    canvas.Update()
    for format in "eps", "png", "root":
        canvas.Print(plot_dir+"/tune_x_vs_y."+format)

units = {"energy":"MeV",}

def main(fglob):
    for file_name in glob.glob(fglob):
        plot_dir = os.path.split(file_name)[0]+"/plot_tune/"
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)
        try:
            data = load_file(file_name)
        except IndexError:
            continue
        plot_data_1d(data, 'fractional cell tune', plot_dir, None, 1, False) # cell tune or ring tune
        plot_data_2d(data, 'fractional cell tune', plot_dir) # cell tune or ring tune, plot x tune vs y tune

if __name__ == "__main__":
    main(sys.argv[1])
    input("Press <CR> to end")
