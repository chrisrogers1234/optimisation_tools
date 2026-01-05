"""
Script to find the closed orbit; drives xboa EllipseClosedOrbitFinder algorithm
"""


import time
import glob
import numpy
import sys
import os
import json
sys.path.insert(1, "scripts")
from opal_tracking import OpalTracking
import xboa.common as common
from xboa.hit import Hit
from xboa.algorithms.closed_orbit import EllipseClosedOrbitFinder

import ROOT

import optimisation_tools.plotting.plot_orbit
from optimisation_tools.utils import utilities

CONFIG = None
OUT_DIR = None
RUN_DIR = None

def reference(energy):
    """
    Generate a reference particle
    """
    hit_dict = {}
    hit_dict["pid"] = CONFIG.tracking["pdg_pid"]
    hit_dict["mass"] = common.pdg_pid_to_mass[abs(hit_dict["pid"])]
    hit_dict["charge"] = 1
    hit_dict["x"] = 0.
    hit_dict["kinetic_energy"] = energy
    return Hit.new_from_dict(hit_dict, "pz")

def plot_iteration(sub_index, i, iteration, energy):
    """
    Plot the closed orbit ellipse and the ellipse fit
    """
    canvas, hist, graph, fit = iteration.plot_ellipse("x", "px", "mm", "MeV/c")
    hist.SetTitle('KE='+str(energy)+' iter='+str(i))
    canvas.Update()
    name = os.path.join(OUT_DIR, "find_closed_orbits")
    name = os.path.join(name, "closed_orbit-sub-index_"+str(sub_index)+"-i_"+str(i))
    for format in "eps", "root", "png":
        canvas.Print(name+"."+format)

def find_closed_orbit(sub_index, subs, seed, config):
    """
    Find the closed orbit; algorithm is to track turn by turn; fit an ellipse to
    the tracking; find the centre of the ellipse; repeat until no improvement or
    10 iterations.
    - energy: (float) kinetic energy at which the co is calculated
    - step: (float) step size in tracking
    - poly_order: (int) order of the polynomial fit to the field map (not used)
    - smooth_oder: (int) order of smoothing the polynomial fit to the field map
                   (not used)
    - seed: (list of 2 floats) [x, px] value to be used as the seed for the next 
            iteration; px value is ignored, sorry about that.
    """
    max_iterations = config.find_closed_orbits["max_iterations"]
    probe = config.find_closed_orbits["probe_files"]
    for key in sorted(subs.keys()):
        print(utilities.sub_to_name(key), subs[key], end=' ')
    print()
    out_dir = OUT_DIR
    run_dir = RUN_DIR
    tmp_dir = "./"
    try:
        os.makedirs(run_dir)
    except OSError: # maybe the dir already exists
        pass
    os.chdir(run_dir)
    print("Running in", os.getcwd())
    common.substitute(CONFIG.tracking["lattice_file"], tmp_dir+'SectorFFAGMagnet.tmp', subs)
    energy = subs["__energy__"]
    ref_hit = reference(energy)
    opal_exe = os.path.expandvars("${OPAL_EXE_PATH}/opal")
    tracking = OpalTracking(tmp_dir+'/SectorFFAGMagnet.tmp', tmp_dir+'/disttest.dat', ref_hit, probe, opal_exe, tmp_dir+"/log")
    seed_hit = ref_hit.deepcopy()
    seed_hit["x"] = seed[0]
    seed_hit["px"] = seed[1]
    # fix momentum
    seed_hit["pz"] = (ref_hit["p"]**2-seed_hit["px"]**2)**0.5
    print("Reference kinetic energy:", ref_hit["kinetic_energy"])
    print("Seed kinetic energy:     ", seed_hit["kinetic_energy"])
    finder = EllipseClosedOrbitFinder(tracking, seed_hit)
    generator = finder.find_closed_orbit_generator(["x", "px"], 1)
    x_std_old = 1e9
    i = -1
    will_loop = True
    iteration = None
    while will_loop:
        try:
            iteration = next(generator)
        except StopIteration:
            will_loop = False
            print(sys.exc_info()[1])
        i += 1
    #for i, iteration in enumerate(generator):
        #for point in iteration.points:
        heading = ['station', 't', 'x', 'px', 'y', 'py', 'z', 'pz', 'r', 'pt', 'kinetic_energy']
        for key in heading:
            print(str(key).rjust(10), end=' ')
        print()
        for hit in tracking.last[0]:
            for key in heading:
                print(str(round(hit[key], 1)).rjust(10), end=' ')
            print()
        if iteration == None:
            continue
        print(iteration.centre)
        #if iteration.centre != None: #i == 0 and 
        if i == 0:
            plot_iteration(sub_index, i, iteration, energy)
        if i >= max_iterations:
            break
        x_mean = numpy.mean([point[0] for point in iteration.points])
        x_std = numpy.std([point[0] for point in iteration.points])
        print("Seed:", iteration.points[0][0], "Mean:", x_mean, "Std:", x_std)
        if type(iteration.centre) != type(None) and x_std >= x_std_old: # require convergence
            break
        x_std_old = x_std
    os.chdir(out_dir)
    if i > 0:
        plot_iteration(sub_index, i, iteration, energy)
    return tracking.last[0]

def get_seed(config, results, subs):
    if len(results) == 0:
        config_seed = config.find_closed_orbits["seed"].pop(0)
        print("Using config seed", config_seed)
        return config_seed, len(config.find_closed_orbits["seed"]) > 0
    axis_candidates = utilities.get_substitutions_axis(results)
    if len(axis_candidates) == 0:
        co_hit = results[0]["hits"][0]
        seed = [co_hit["x"], co_hit["px"]]
        print("Using last co as seed", seed)
        return seed, False
    print("Doing interpolation for seed for variables", list(axis_candidates.keys()))
    # [dx, dpx]
    delta_keys_new = {}
    delta_var_new = []
    seed_var = {"x":results[-1]["hits"][0]["x"], "px":results[-1]["hits"][0]["px"]}
    print("  base", seed_var)
    for key in list(axis_candidates.keys()):
        key2 = subs[key]
        key1 = axis_candidates[key][-1]
        key0 = axis_candidates[key][-2]
        print("    key", key, "values", key0, key1, key2)
        for var in list(seed_var.keys()):
            var1 = results[-1]["hits"][0][var]
            var0 = results[-2]["hits"][0][var]
            if abs(key1-key0) < 1e-9: # div0
                seed_var[var] = var1
                continue
            delta = (var1-var0)/(key1-key0)*(key2-key1)
            print("      var", var, "values", var1, var0, "delta", delta)
            seed_var[var] += delta
    print("  seed", seed_var)
    seed = [seed_var["x"], seed_var["px"]]
    return seed, False

def main(config):
    global CONFIG, OUT_DIR, RUN_DIR
    CONFIG = config
    OUT_DIR = CONFIG.run_control["output_dir"]
    utilities.clear_dir(os.path.join(OUT_DIR, "find_closed_orbits"))

    RUN_DIR = os.path.join(OUT_DIR, "tmp/find_closed_orbits/")
    fout_name = os.path.join(OUT_DIR, config.find_closed_orbits["output_file"])
    fout = open(fout_name+".tmp", 'w')
    subs_list = config.substitution_list
    results = []
    for i, sub in enumerate(subs_list):
        will_loop = True
        is_batch = i >= config.find_closed_orbits["root_batch"] 
        for item, key in config.find_closed_orbits["subs_overrides"].items():
            sub[item] = key
        ROOT.gROOT.SetBatch(is_batch)
        hit_list = []
        while will_loop:
            try:
                next_seed, will_loop = get_seed(config, results, sub)
            except Exception:
                co_hit = results[0]["hits"][0]
                next_seed = [co_hit["x"], co_hit["px"]]
                will_loop = False
            print("LOOPING DEBUG", next_seed, will_loop, sub['__energy__'])
            try:
                hit_list = find_closed_orbit(i, sub, next_seed, config)
            except IndexError as ValueError:
                sys.excepthook(*sys.exc_info())
            except RuntimeError:
                sys.excepthook(*sys.exc_info())
                hit_list = []
                print("Breaking loop due to tracking error")
                will_loop = False
            will_loop = len(hit_list) < 20 and will_loop
        output = {"substitutions":sub, "hits":[hit.dict_from_hit() for hit in hit_list]}
        results.append(output)
        print(json.dumps(output), file=fout)
        fout.flush()
        if config.find_closed_orbits["do_plot_orbit"] and i == 0:
            plot_orbit.main(config.run_control["output_dir"],
                            "tmp/find_closed_orbits/",
                            "SectorFFAGMagnet-trackOrbit.dat")
    fout.close()
    time.sleep(1)
    os.rename(fout_name+".tmp", fout_name+".out")
    ROOT.gROOT.SetBatch(False)
    #disabled because I need to hack around file names (thats all)
    #if len(energy_list) < 5:
    #    plot_orbit.main()
    #    print "Finished closed orbits"

