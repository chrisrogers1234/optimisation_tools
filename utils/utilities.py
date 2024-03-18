import math
import subprocess
import copy
import os
import shutil
import io
import sys
import importlib
import json

import numpy
import ROOT

import xboa.common
import xboa.hit


from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.opal_tracking import G4BLTracking
from optimisation_tools.opal_tracking import StoreDataInMemory
from optimisation_tools.opal_tracking import PyOpalTracking
from optimisation_tools.opal_tracking import PyOpalTracking2

import matplotlib
import matplotlib.pyplot

def setup_large_figure(axes, f_size = 20, l_size = 14):
    axes.set_xlabel(axes.get_xlabel(), fontsize=f_size)
    axes.set_ylabel(axes.get_ylabel(), fontsize=f_size)
    legend = axes.get_legend()
    if legend:
        legend.fontsize = l_size
        legend.prop.set_size(f_size)
    axes.tick_params(labelsize = l_size)


def sub_to_name(sub_key):
    sub_name = sub_key[2:-2]
    sub_name = sub_name.replace("_", " ")
    sub_name = sub_name[0].upper()+sub_name[1:]
    return sub_name

def sub_to_units(sub_key):
    units = {
        "energy":"MeV"
    }
    name = sub_to_name(sub_key).lower()
    if name in units:
        return " ["+units[name]+"]"
    return ""

def clear_dir(a_dir):
    try:
        shutil.rmtree(a_dir)
    except OSError:
        pass
    os.makedirs(a_dir)

def preprocess_subs(subs):
    subs_tmp = copy.deepcopy(subs)
    for key, value in subs_tmp.items():
        if type(value) == type([]):
            list_str = "{"
            for list_item in value[:-1]:
                list_str += str(list_item)+", "
            list_str += str(value[-1])+"}"
            subs[key] = list_str
        elif value is True:
            subs[key] = "True" #"TRUE"
        elif value is False:
            subs[key] = "False" #"FALSE"
    return subs

def dict_compare(list_of_dicts, include_missing=True, float_tolerance=1e-9):
    """
    Return a set of keys that are not the same for all dicts in the list
    - list_of_dicts: list of dicts that are almost the same
    - include_missing: if True, include keys when they are missing from one or 
      more of the dicts
    - float_tolerance: tolerance for float comparisons; where floats differ 
      by more than float_tolerance then count them as different. Note that 
      the float comparison is only done at the top level (so floats in sub-dicts 
      are only checked for difference)
    Return a set of keys. Keys will be included if they are not present, or 
    differ by the small amount
    """
    delta_keys = []
    dict_1 = list_of_dicts[0]
    for dict_2 in list_of_dicts[1:]:
        for key in dict_1:
            if key not in dict_2:
                if include_missing:
                    delta_keys.append(key)
                continue
            if dict_2[key] == dict_1[key]:
                continue
            if float_tolerance and type(dict_1[key]) == type(1.0) and type(dict_2[key]) == type(1.0):
               if abs(dict_1[key] - dict_2[key]) < float_tolerance:
                    continue
            delta_keys.append(key)
    return set(delta_keys)

def do_lattice(config, subs, overrides, hit_list = None, tracking = None):
    subs = copy.deepcopy(subs)
    subs.update(overrides)
    if hit_list:
        # WARNING - ignores min_track_number in opal_tracking
        subs.update({"__n_particles__":len(hit_list)})
    if not will_do_python(config):
        subs = preprocess_subs(subs)
    lattice_in = config.tracking["lattice_file"]
    lattice_out = config.tracking["lattice_file_out"]
    if type(lattice_in) == type(""):
        lattice_in = [lattice_in] 
    if type(lattice_out) == type(""):
        lattice_out = [lattice_out] 
    if len(lattice_out) != len(lattice_in):
        raise ValueError(f"lattice in length did not match lattice out length {lattice_in}, {lattice_out}")
    for lin, lout in zip(lattice_in, lattice_out):
        if will_do_python(config):
            xboa.common.substitute(lin, lout, {})
            if tracking == None:
                raise ValueError("Tracking was None but a value is needed when using python")
            tracking.overrides = subs
        else:
            xboa.common.substitute(lin, lout, subs)
    out_dir = os.path.dirname(lattice_out[0])
    with open(os.path.join(out_dir, "subs.json"), "w") as out_file:
        out_file.write(json.dumps(subs, indent=2))
    return subs

def reference(config, energy, x=0., px=0., y=0., py=0.):
    """
    Generate a reference particle
    """
    hit_dict = {}
    hit_dict["pid"] = config.tracking["pdg_pid"]
    hit_dict["mass"] = xboa.common.pdg_pid_to_mass[abs(hit_dict["pid"])]
    hit_dict["charge"] = 1
    hit_dict["x"] = x
    hit_dict["px"] = px
    hit_dict["y"] = y
    hit_dict["py"] = py
    hit_dict["kinetic_energy"] = energy*1e3
    hit = xboa.hit.Hit.new_from_dict(hit_dict, "pz")
    return hit

def will_do_python(config):
    return config.tracking["opal_path"] == "python"

def setup_tracking(config, probes, ref_energy):
    if config.tracking["opal_path"] == None:
        return setup_tracking_g4bl(config, probes, ref_energy)
    if will_do_python(config):
        return setup_py_tracking_2(config, probes, ref_energy)
    ref_hit = reference(config, ref_energy)
    opal_exe = os.path.expandvars(config.tracking["opal_path"])
    lattice = config.tracking["lattice_file_out"]
    if type(lattice) == type([]):
        lattice = lattice[0]
    log = config.tracking["tracking_log"]
    beam = config.tracking["beam_file_out"]
    will_do_mpi = config.tracking["will_do_mpi"]
    tracking = OpalTracking(lattice, beam, ref_hit, probes, opal_exe, log, 1, will_do_mpi)
    tracking.verbose = config.tracking["verbose"]
    tracking.set_file_format(config.tracking["file_format"])
    tracking.flags = config.tracking["flags"]
    tracking.pass_through_analysis = StoreDataInMemory(config)
    return tracking

def setup_tracking_g4bl(config, outfiles, ref_energy):
    ref_hit = reference(config, ref_energy)
    g4bl_exe = os.path.expandvars(config.tracking["g4bl_path"])
    lattice = config.tracking["lattice_file_out"]
    log = config.tracking["tracking_log"]
    beam = config.tracking["beam_file_out"]
    tracking = G4BLTracking(lattice, beam, ref_hit, outfiles, g4bl_exe, log)
    tracking.beam_format = config.tracking["g4bl_beam_format"]
    tracking.output_format = config.tracking["g4bl_output_format"]
    tracking.verbose = config.tracking["verbose"]
    tracking.flags = config.tracking["flags"]
    tracking.clear_path = config.tracking["clear_files"]
    tracking.pass_through_analysis = StoreDataInMemory(config)
    return tracking

PY_OPAL_TRACKING = None
def setup_py_tracking(config, run_dir, phi_list):
    global PY_OPAL_TRACKING
    if PY_OPAL_TRACKING is not None:
        print("Warning - tried to setup PyOpal again - abort")
        return PY_OPAL_TRACKING
    lattice = config.tracking["lattice_file_out"]
    tracking = PyOpalTracking(config, run_dir)
    tracking.step_list = phi_list
    PY_OPAL_TRACKING = tracking
    return tracking

def setup_py_tracking_2(config, probes, ref_energy):
    ref_hit = reference(config, ref_energy)
    tracking = PyOpalTracking2(config, probes, ref_hit)
    tracking.pass_through_analysis = StoreDataInMemory(config)
    tracking.set_file_format(config.tracking["file_format"])
    return tracking


def tune_lines(canvas, min_order=0, max_order=8):
    canvas.cd()
    for x_power in range(0, max_order):
        for y_power in range(0, max_order):
            if y_power + x_power > max_order or y_power + x_power == 0:
                continue
            x_points = [0., 1.]
            y_points = [0., 1.]
            if x_power > y_power:
                x_points[0] = y_points[0]*y_power/x_power
                x_points[1] = y_points[1]*y_power/x_power
            else:
                y_points[0] = x_points[0]*x_power/y_power
                y_points[1] = x_points[1]*x_power/y_power
            hist, graph = xboa.common.make_root_graph("", x_points, "", y_points, "")
            graph.Draw("SAMEL")
            x_points = [1.-x for x in x_points]
            #y_points = [1.-y for y in y_points]
            hist, graph = xboa.common.make_root_graph("", x_points, "", y_points, "")
            graph.Draw("SAMEL")
    canvas.Update()

def get_substitutions_axis(data, subs_key):
    subs_ref = data[0][subs_key]
    axis_candidates = {}
    for item in data:
        subs = item[subs_key]
        for key in list(subs.keys()):
            try:
                comp = subs[key] == subs_ref[key]
            except KeyError:
                print("Warning - missing ", key, "treating as same")
                comp = True
            if not comp:
                try:
                    float(subs[key])
                    axis_candidates[key] = []
                except (TypeError, ValueError):
                    continue
    if axis_candidates == {}:
        print("All of", len(data), "items look the same - nothing to plot")
        print("First:")
        print(" ", data[0][subs_key])
        print("Last:")
        print(" ", data[-1][subs_key])
    for item in data:
        for key in axis_candidates:
            #print key, axis_candidates.keys()
            #print item.keys()
            #print item[subs_key].keys()
            axis_candidates[key].append(item[subs_key][key])
    return axis_candidates

def get_groups(data, group_axis, subs_key):
    axis_candidates = get_substitutions_axis(data, subs_key)
    if group_axis != None and group_axis not in axis_candidates:
        raise RuntimeError("Did not recognise group axis "+str(group_axis))
    if group_axis == None:
        group_list = [item for item in data]
        return {"":{"item_list":[i for i in range(len(data))]}}
    else:
        # group_list is lists the possible groups
        # e.g. list of all possible values of "__bump_field_1__"
        group_list = [item[subs_key][group_axis] for item in data]
        group_list = list(set(group_list)) # unique list
        # tmp_group_dict is mapping from group value to the items having that value
        tmp_group_dict = dict([(group, []) for group in group_list])
        for key in tmp_group_dict:
            tmp_group_dict[key] = [i for i, item in enumerate(data) \
                                        if item[subs_key][group_axis] == key]
    group_dict = {}
    for key in tmp_group_dict:
        new_key = sub_to_name(group_axis)+" "+format(key, "3.3g")
        group_dict[new_key] = {'item_list':tmp_group_dict[key]}
        print(new_key, ":", group_dict[new_key])
    return group_dict

def matplot_marker_size(x_points):
    """Get the marker size using number of points; x_points can be the number or a list"""
    marker_size = 1
    if type(x_points) == type(1):
        x_length = x_points
    else:
        x_length = len(x_points)
    if x_length:
        marker_size = 100/x_length**0.5
    return marker_size

def setup_gstyle():
    stops = [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000]
    red   = [0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764]
    green = [0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832]
    blue  = [0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539]
    s = numpy.array(stops)
    r = numpy.array(red)
    g = numpy.array(green)
    b = numpy.array(blue)

    ncontours = 255
    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)

    # axes and labels
    ROOT.gStyle.SetPadBottomMargin(0.15)
    ROOT.gStyle.SetPadLeftMargin(0.15)
    ROOT.gStyle.SetPadRightMargin(0.15)
    for axis in "X", "Y":
        ROOT.gStyle.SetNdivisions(505, axis)
        ROOT.gStyle.SetLabelSize(0.05, axis)
        ROOT.gStyle.SetTitleSize(0.06, axis)
        ROOT.gStyle.SetTitleOffset(1.10, axis)


def setup_da_figure_single(include_projections):
    fig = matplotlib.pyplot.figure()
    y, dy = 0.11, 0.34
    if include_projections:
        y, dy = 0.20, 0.30
    axes = [
        fig.add_subplot(2, 1, 1,  position=[0.11, 0.56, 0.86, 0.34]),
        fig.add_subplot(2, 6, 2,  position=[0.11, y, 0.36, dy]),
        fig.add_subplot(2, 6, 3,  position=[0.61, y, 0.36, dy]),
    ]

    if include_projections:
        axes += [
            fig.add_subplot(2, 7, 4,  position=[0.06, 0.10, 0.38, 0.05]),
            fig.add_subplot(2, 7, 5,  position=[0.56, 0.10, 0.38, 0.05]),
        ]
    return fig, axes

def setup_da_figure_decoupling(include_projections):
    fig = matplotlib.pyplot.figure(figsize=(20, 10))
    y, dy = 0.10, 0.35
    if include_projections:
        y, dy = 0.15, 0.30
    axes = [
        fig.add_subplot(2, 3, 1,  position=[0.06, 0.55, 0.26, 0.35]),
        fig.add_subplot(2, 6, 7,  position=[0.06, y, 0.10, dy]),
        fig.add_subplot(2, 6, 8,  position=[0.22, y, 0.10, dy]),
        fig.add_subplot(2, 3, 2,  position=[0.38, 0.55, 0.26, 0.35]),
        fig.add_subplot(2, 6, 9,  position=[0.38, 0.10, 0.10, 0.35]),
        fig.add_subplot(2, 6, 10, position=[0.54, 0.10, 0.10, 0.35]),
        fig.add_subplot(2, 3, 3,  position=[0.70, 0.55, 0.26, 0.35]),
        fig.add_subplot(2, 6, 11, position=[0.70, 0.10, 0.10, 0.35]),
        fig.add_subplot(2, 6, 12, position=[0.86, 0.10, 0.10, 0.35]),
    ]

    if include_projections:
        axes += [
            fig.add_subplot(2, 7, 13,  position=[0.06, 0.10, 0.10, 0.05]),
            fig.add_subplot(2, 7, 14,  position=[0.22, 0.10, 0.10, 0.05]),
        ]
    return fig, axes


def setup_da_figure_regular(include_projections):
    fig = matplotlib.pyplot.figure(figsize=(20, 10))
    y, dy = 0.10, 0.35
    if include_projections:
        y, dy = 0.15, 0.30
    axes = [
        fig.add_subplot(2, 2, 1,  position=[0.06, 0.55, 0.42, 0.35]),
        fig.add_subplot(2, 4, 5,  position=[0.06, y, 0.18, dy]),
        fig.add_subplot(2, 4, 6,  position=[0.30, y, 0.18, dy]),
        fig.add_subplot(2, 2, 2,  position=[0.56, 0.55, 0.42, 0.35]),
        fig.add_subplot(2, 4, 7, position=[0.56, 0.10, 0.18, 0.35]),
        fig.add_subplot(2, 4, 8, position=[0.80, 0.10, 0.18, 0.35]),
    ]

    if include_projections:
        axes += [
            fig.add_subplot(2, 7, 13,  position=[0.06, 0.10, 0.10, 0.05]),
            fig.add_subplot(2, 7, 14,  position=[0.22, 0.10, 0.10, 0.05]),
        ]
    return fig, axes


def plot_id(figure, lattice, study, version, x=0.9, y=0.95):
    text = lattice.replace("_", " ")
    text += "\n"+study.replace("_", " ")
    if version != "release":
        text += " "+str(version)
    else:
        text += " "+str(version)
    figure.text(x, y, text, fontsize=10)

def directory_name(lattice, study, version):
    dir_name = lattice.replace(" ", "_")+"_"+study.replace(" ", "_")
    if version == "release":
        dir_name += "r"
    else:
        dir_name += "_"+str(version)
    dir_name += "/"
    return dir_name

def get_config(path_code="scripts/"):
    if len(sys.argv) < 2:
        print("Usage: python /path/to/run_one.py /path/to/config.py")
        sys.exit(1)
    config_path, config_file_name = os.path.split(sys.argv[1])
    print(config_path, config_file_name)
    config_module = config_file_name.replace(".py", "")
    sys.path.append(config_path)
    config_args = tuple(sys.argv[2:])
    print("Using configuration module", config_module, "with arguments", config_args)
    config_mod = importlib.import_module(config_module)
    config = config_mod.Config(*config_args)
    return sys.argv[1], config

def mencode(working_directory, fin_str, fout_name):
    here = os.getcwd()
    os.chdir(working_directory)
    print("Making movie in", os.getcwd())
    try:
        output = subprocess.check_output(["mencoder",
                                "mf://"+fin_str,
                                "-mf", "w=800:h=600:fps=5:type=png",
                                "-ovc", "lavc",
                                "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                "-oac", "copy",
                                "-o", fout_name])
    except:
        sys.excepthook(*sys.exc_info())
        print("Movie failed")
    finally:
        os.chdir(here)

def scale(hit_in, k, tan_delta, p_out):
    """
    Find a hit in the scaled coordinate system of p_out where p_out is the new total momentum

    Note this is complicated because we have to apply a scale in radius and momentum but also
    angle because the spiral angle will change for constant physical angle.
    """
    # following spiral_sector.pdf
    p_in = hit_in["p"]
    beta = p_out / p_in
    alpha = beta**(1/(k+1))
    hit_out = copy.deepcopy(hit_in)
    # assume y is vertical, x and z is the plane of the FFA ring
    hit_out["y"] = alpha*hit_in["y"]

    phi_in = math.atan2(hit_in["z"], hit_in["x"])
    phi_out = phi_in + math.log(alpha)*tan_delta
    r_in = (hit_in["x"]**2+hit_in["z"]**2)**0.5
    hit_out["x"] = r_in*alpha*math.cos(phi_out)
    hit_out["z"] = r_in*alpha*math.sin(phi_out)

    hit_out["p"] = beta*hit_in["p"]
    p_in = (hit_in["px"]**2+hit_in["pz"]**2)**0.5
    px, pz = hit_out["px"], hit_out["pz"]
    hit_out["px"] = px*math.cos(phi_out) - pz*math.sin(phi_out)
    hit_out["pz"] = pz*math.cos(phi_out) + px*math.sin(phi_out)
    #hit_out["px"] = +hit_in["px"]*beta*math.cos(phi_out)-hit_in["pz"]*beta*math.sin(phi_out)
    #hit_out["pz"] = -hit_in["px"]*beta*math.sin(phi_out)+hit_in["pz"]*beta*math.cos(phi_out)
    hit_out.mass_shell_condition("energy")

    print("scale alpha", alpha, "beta", beta)

    return hit_out

def scale_2(x_in, xp_in, ke_in, ke_out, k, tan_delta, pid = 2212):
    """
    Returns tuple of x, xp
    """
    hit = xboa.hit.Hit.new_from_dict({"pid":pid, "mass":xboa.common.pdg_pid_to_mass[2212], "kinetic_energy":ke_in, "x":x_in, "px":0, "py":0, "pz":0})
    hit.mass_shell_condition("pz")
    hit["x'"] = xp_in
    hit.mass_shell_condition("p")
    print(xp_in, hit["x'"])
    print(ke_in, hit["kinetic_energy"])
    p_out = ((hit["mass"]+ke_out)**2-hit["mass"]**2)**0.5
    hit_out = scale(hit, k, tan_delta, p_out)
    print(ke_out, hit_out["kinetic_energy"])
    print(hit_out["x"], hit_out["x'"])
    return hit_out["x"], hit_out["x'"]


