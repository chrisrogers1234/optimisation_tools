"""
Plot a single closed orbit (once tracking has finished)
"""
import sys
import copy
import os
import math
import argparse
import glob
import shutil

import matplotlib
import ROOT

import xboa.common

import optimisation_tools.utils.utilities
import optimisation_tools.plotting.plot as plot
import pyopal.objects.parser
import pyopal.objects.track_run

def pybase():
    base_dir = "output/2022-12-01_baseline/baseline/"
    run_dir = "tmp/find_closed_orbits"
    name = "pybase"
    orbit_folder_list = [os.path.join(base_dir, "tmp/find_closed_orbits")]
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "fets_ffa"


def closed_orbits_k_vary():
    base_dir = "output/2022-07-01_baseline/bump_quest_v8/"
    by = "by=0.00"
    name = "closed_orbits"
    run_dir = name+"_k=6.1/tmp/find_closed_orbits_3.0"
    orbit_folder_list = glob.glob(base_dir+name+"_k=*/tmp/find_closed_orbits_*")
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "FETS_Ring"

def find_bump():
    base_dir = "output/2022-07-01_baseline/bump_quest_v10/"
    by = "by=0.10"
    name = "find_bump"
    run_dir = name+"_r0=-000_"+by+"_k=8.0095/tmp/find_bump"
    orbit_folder_list = glob.glob(base_dir+name+"_r0=-*_"+by+"_k=8.0095/tmp/find_bump")
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "FETS_Ring"

def track_bump():
    base_dir = "output/2022-07-01_baseline/bump_quest_v10/"
    by = "by=0.00"
    a_dir = "track_bump_r0=-*_"+by+"_k=8.0095/"
    inj_dir = "track_bump_r0=-100_"+by+"_k=8.0095/"
    run_dir = inj_dir+"/track_beam/reference"
    off_ref_dir = "output/2022-07-01_baseline/bump_quest_v8/closed_orbits_k=6.1/tmp/find_closed_orbits_3.0"
    orbit_folder_list = glob.glob(base_dir+a_dir+"/track_beam/reference")+\
                        glob.glob(base_dir+inj_dir+"/track_beam/injection")+\
                        [off_ref_dir]
    print(orbit_folder_list)
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range, "track_bump_"+by, "FETS_Ring"

def track_ffynchrotron_folders():
    a_dir = "output/muon_ffynchrotron/baseline/"
    base_dir = a_dir+""
    run_dir = "tmp/find_closed_orbits"
    orbit_folder_list = glob.glob(a_dir+"/"+run_dir)
    plot_range = [13, 17]
    return base_dir, run_dir, orbit_folder_list, plot_range, "ffynchrotron_bump_co", "Ffynchrotron"

def find_ffynchrotron_folders():
    a_dir = "output/muon_ffynchrotron/bump_quest_v15/"
    base_dir = a_dir+"energy=800/"
    run_dir = "tmp/find_bump"
    orbit_folder_list = glob.glob(a_dir+"energy=*/"+run_dir)
    plot_range = [90, 150]
    return base_dir, run_dir, orbit_folder_list, plot_range, "ffynchrotron_bump_co", "Ffynchrotron"

def generate_name(orbit_folder):
    return ""
    orbit_folder = orbit_folder.split("_k=")[1]
    orbit_folder = orbit_folder.split("/")[0]
    return "k="+orbit_folder

def plot_azimuthal():
    base_dir, run_dir, orbit_folder_list, plot_range, job_name, lattice_name = pybase()
    r0 = 4.0
    orbit_folder_list = [(o, generate_name(o)) for o in orbit_folder_list]
    probe_files = "*PROBE*.h5" #"FOILPROBE_1.h5" #
    track_files = "*-trackOrbit*dat"
    lattice_file = os.path.join(base_dir, run_dir+"/"+lattice_name+".tmp")
    angle_domain = [0.0, 360.0]
    allowed_events = ["ID1"]
    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    test_function = lambda words: words[1]**2+words[3]**2 > (4.4)**2# or words[1]**2+words[3]**2 < (3.5)**2
    orbit_list = []
    for orbit_folder, name in orbit_folder_list:
        h5 = plot.LoadH5(os.path.join(orbit_folder, probe_files))
        print(os.path.join(orbit_folder, track_files))
        try:
            orbit_file = glob.glob(os.path.join(orbit_folder, track_files))[0]
        except IndexError:
            print("Failed to find orbit file on", orbit_folder)
            continue
        print(orbit_folder, track_files, orbit_file)
        orbit = plot.LoadOrbit(orbit_file, allowed_events, test_function)
        orbit_list.append(plot.PlotOrbit(orbit, name, h5))
        orbit_list[-1].r0 = r0
        print("Loaded", orbit_file)
    if not len(orbit_list):
        raise RuntimeError("Could not find any orbit files", orbit_file)
    print("BASE_DIR", base_dir, "LATTICE FILE", lattice_file)
    plot_fields = plot.PlotFields(plot.GetFields(lattice_file), orbit_list)
    #[print([plot_orbit.orbit.orbit[var][0] for var in ["px", "py", "pz"]]) for plot_orbit in orbit_list]
    plot_fields.azimuthal_field_plot = ["bz", "div_b", "curl_b"]
    figure = plot_fields.azimuthal_fig("bump", plot_range)
    for axes in figure.axes:
        if "$r_0\\phi$" in axes.get_xlabel():
            continue
        axes.grid(True)
    figname = os.path.join(base_dir, job_name+".png")
    figure.savefig(figname)
    print("Figure saved as", figname)

def plot_beam():
    base_dir, run_dir, orbit_folder_list, plot_range, job_name, lattice_name = pybase()
    rmax = 5.0
    b0 = 1.0
    orbit_folder_list = [(orbit, generate_name(orbit)) for orbit in orbit_folder_list]
    probe_files = "*PROBE*.h5" #"FOILPROBE_1.h5" #
    lattice_file = os.path.join(base_dir, run_dir+"/run_"+lattice_name+".py")
    log_file = os.path.join(base_dir, run_dir+"log")
    angle_domain = [-0.0, 360.0]
    allowed_events = None #["ID1"]
    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    test_function = lambda words: words[1]**2+words[3]**2 > (rmax)**2
    orbit_list = []
    #log_file = plot.LoadLog(log_file)
    #log_file.element_lambda = lambda element: "MAGNET" in element["name"] and "BUMP" in element["name"]
    #log_file.print()
    for orbit_folder, name in orbit_folder_list:
        h5 = plot.LoadH5(os.path.join(orbit_folder, probe_files))
        orbit_file = os.path.join(orbit_folder, lattice_name+"-trackOrbit.dat")
        if not os.path.exists(orbit_file):
            orbit_file = os.path.join(orbit_folder, lattice_name+"-trackOrbit_1.dat")
        orbit = plot.LoadOrbit(orbit_file, allowed_events, test_function)
        orbit_list.append(plot.PlotOrbit(orbit, name, h5))
    plot_fields = plot.PlotFields(plot.GetFields(lattice_file), orbit_list)
    plot_fields.polygon_plotter = plot.PlotPolygon(n_cells=16, cell_length=1.7)
    #plot_fields.log_plotter = log_file
    plot_fields.n_2d_points = 500
    plot_fields.b0 = b0
    #figure = plot_fields.field_fig("baseline", [0., 0., 0., 0.], [-5, 5], [-5, 5], False)
    #figure.savefig(os.path.join(base_dir, job_name+"_fields2d.png"))

    figure = plot_fields.one_field_fig("baseline", [0., 0., 0., 0.], [-rmax, rmax], [-rmax, rmax], "bz")
    figure.suptitle("")
    path = os.path.join(base_dir, "../"+job_name+"_fields_bz.png")
    figure.savefig(path)
    print("saved to", path)

def main():
    #test()
    plot_beam()
    #plot_azimuthal()
    #plot_rf()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("Press <CR> to finish")

