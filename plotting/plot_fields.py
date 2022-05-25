"""
Plot a single closed orbit (once tracking has finished)
"""
import sys
import copy
import os
import math
import argparse
import h5py
import glob
import shutil


import matplotlib
import ROOT

import xboa.common

import PyOpal.parser
import PyOpal.field
import optimisation_tools.utils.utilities
import optimisation_tools.plotting.plot as plot

def ramp_folders():
    run_dir = "tmp/track_beam/"
    base_dir = "output/2022-03-01_baseline/correlated_painting/tracking/"
    orbit_folder_list = [base_dir+"/track_beam/"+f for f in ["plotting", "reference", "injected_beam"]]
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range

def find_folders():
    a_dir = "output/2022-03-01_baseline/correlated_painting/bump_quest_v16/"
    base_dir = sorted(glob.glob(a_dir+"find_bump_th_090_r0_-*"))[0]
    run_dir = "tmp/find_bump/"
    orbit_folder_list = glob.glob(a_dir+"/find_bump_th_090_r0_-*/tmp/find_bump/") #[(base_dir+run_dir, "Horizontal FFA"),]
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range

def track_folders():
    a_dir = "output/2022-03-01_baseline/correlated_painting/bump_quest_v11/"
    base_dir = a_dir+"/track_bump_th_090_r0_-000/"
    #sorted(glob.glob(a_dir+"track_bump_th_090_r0_-*"))[0]
    run_dir = "track_beam/reference/"
    orbit_folder_list = [base_dir+run_dir]+glob.glob(a_dir+"/track_bump_th_090_r0_-*/"+run_dir)
    for folder in ["injection"]:
        orbit_folder_list += [base_dir+"track_beam/"+folder]
    plot_range = [65, 90]
    return base_dir, run_dir, orbit_folder_list, plot_range

def co_folders():
    a_dir = "output/2022-03-01_baseline/baseline"
    base_dir = a_dir+""
    run_dir = "tmp/find_closed_orbits"
    orbit_folder_list = [base_dir+"tmp/find_closed_orbits"] #[base_dir+run_dir]+glob.glob(a_dir+"/track_bump_th_090_r0_-*/"+run_dir)
    plot_range = [0, 360]
    return base_dir, run_dir, orbit_folder_list, plot_range


def track_ramp_folders():
    a_dir = "output/2022-03-01_baseline/correlated_painting/tracking_v4/"
    base_dir = a_dir+""
    run_dir = "track_beam/injected_beam"
    orbit_folder_list = [] #[base_dir+run_dir]+glob.glob(a_dir+"/track_bump_th_090_r0_-*/"+run_dir)
    for folder in ["injected_beam"]:
        orbit_folder_list += [base_dir+"track_beam/"+folder]
    plot_range = [0, 360]
    return base_dir, run_dir, orbit_folder_list, plot_range

def track_ffynchrotron_folders():
    a_dir = "output/muon_ffynchrotron/"
    base_dir = a_dir+""
    run_dir = "tmp/find_closed_orbits"
    orbit_folder_list = [a_dir+run_dir]
    plot_range = [0, 360]
    return base_dir, run_dir, orbit_folder_list, plot_range

def plot_azimuthal():
    base_dir, run_dir, orbit_folder_list, plot_range = co_folders()
    lattice_name="Ffynchrotron" # "FETS_Ring"
    orbit_folder_list = [(o, "") for o in orbit_folder_list]
    probe_files = "*PROBE*.h5" #"FOILPROBE_1.h5" #
    track_files = "*-trackOrbit*dat"
    lattice_file = os.path.join(base_dir, run_dir+"/"+lattice_name+".tmp")
    angle_domain = [-0.0, 360.0] #[104, 112] #  
    allowed_events = ["ID1"]

    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    test_function = None #lambda words: words[1]**2+words[3]**2 > 4.25**2# or math.atan2(float(words[1]), float(words[3])) > 3.0
    orbit_list = []
    for orbit_folder, name in orbit_folder_list:
        h5 = plot.LoadH5(os.path.join(orbit_folder, probe_files))
        print(os.path.join(orbit_folder, track_files))
        print(os.path.join(orbit_folder, track_files))
        orbit_file = glob.glob(os.path.join(orbit_folder, track_files))[0]
        print(orbit_folder, track_files, orbit_file)
        orbit = plot.LoadOrbit(orbit_file, allowed_events, test_function)
        orbit_list.append(plot.PlotOrbit(orbit, name, h5))
        orbit_list[-1].r0 = 4.0
        print("Loaded", orbit_file)
    lattice_fets = "output/2022-03-01_baseline/baseline/tmp/find_closed_orbits/FETS_Ring.tmp"
    plot_fields = plot.PlotFields(plot.GetFields(lattice_file), orbit_list)
    [print([plot_orbit.orbit.orbit[var][0] for var in ["px", "py", "pz"]]) for plot_orbit in orbit_list]
    job_name = "closed_orbit_bump"
    figure = plot_fields.azimuthal_fig("Bump", plot_range)
    for axes in figure.axes:
        if "$r_0\\phi$" in axes.get_xlabel():
            continue
        axes.grid(True)
    figname = os.path.join(base_dir, job_name+".png")
    figure.savefig(figname)
    print("Figure saved as", figname)

def plot_beam():
    base_dir = "output/muon_ffynchrotron/"
    orbit_folder_list = [(os.path.join(base_dir, "tmp/find_closed_orbits"), "Orbit")]
    lattice_name = "Ffynchrotron"
    probe_files = "*.h5" #"FOILPROBE_1.h5" #
    lattice_file = os.path.join(base_dir, "tmp/find_closed_orbits/{0}.tmp".format(lattice_name))
    log_file = os.path.join(base_dir, "tmp/find_closed_orbits/log")
    angle_domain = [-0.0, 360.0] #[104, 112] #  
    plot_range = [0.0, 360]
    allowed_events = ["ID1"]
    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    test_function = lambda words: False
    orbit_list = []
    log_file = plot.LoadLog(log_file)
    log_file.element_lambda = lambda element: "MAGNET" in element["name"] and "BUMP" in element["name"]
    log_file.print()
    for orbit_folder, name in orbit_folder_list:
        orbit_file = os.path.join(orbit_folder, lattice_name+"-trackOrbit.dat")
        if not os.path.exists(orbit_file):
            orbit_file = os.path.join(orbit_folder, lattice_name+"-trackOrbit_1.dat")
        orbit = plot.LoadOrbit(orbit_file, allowed_events, test_function)
        orbit_list.append(plot.PlotOrbit(orbit, name, None))
    plot_fields = plot.PlotFields(plot.GetFields(lattice_file), orbit_list)
    plot_fields.polygon_plotter = plot.PlotPolygon(25, 11.2)
    plot_fields.log_plotter = log_file
    plot_fields.n_2d_points = 100
    plot_fields.b0 = 20.1
    job_name = "closed_orbit_bump"
    figure = plot_fields.field_fig("baseline", [0., 0., 0., 0.], [900, 1025], [-1100, 1100], False)
    figure.savefig(os.path.join(base_dir, job_name+"_fields2d.png"))
    #figure = matplotlib.pyplot.figure()
    #axes = figure.add_subplot(1, 1, 1)
    #plot_fields.plot_2d(figure, axes, [0.0, 0.0, 0.0, 0.0], [-6, 6], [-6, 6], [-plot_fields.b0, plot_fields.b0], "x", "y", "bz")
    #if plot_fields.polygon_plotter != None:
    #    plot_fields.polygon_plotter.plot(axes)
    #orbit_list[-1].plot_2d(axes, "x", "y", [-5, 5], [-5, 5])
    #axes.set_title("")
    #axes.set_xlabel("x [m]", fontsize=16)
    ##axes.set_ylabel("y [m]", fontsize=16)
    #axes.tick_params(labelsize = 14)
    #matplotlib.pyplot.text(5.10, 5.13, "B$_{z}$ [T]", fontsize=14)
    #figure.savefig(os.path.join(base_dir, job_name+"_fields2d.png"))

def plot_rf():
    base_dir = "output/arctan_baseline/baseline_test_rf_2/"
    orbit_folder_list = [(os.path.join(base_dir, "track_beam_rf_on/grid"), "Orbit")]
    probe_files = "*.h5" #"FOILPROBE_1.h5" #
    lattice_file = os.path.join(base_dir, "track_beam_rf_on/grid/VerticalSectorFFA.tmp")
    log_file = os.path.join(base_dir, "track_beam_rf_on/grid/log")
    angle_domain = [-0.0, 360.0] #[104, 112] #  
    allowed_events = ["ID1"]
    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    t0 = 1151.534795/8.0
    r0 = 4.357
    e0 = 0.015
    test_function = lambda words: False
    orbit_list = []
    log_file = plot.LoadLog(log_file)
    log_file.element_lambda = lambda element: "VARIABLE_RF_CAVITY" in element["name"]
    log_file.print()
    for orbit_folder, name in orbit_folder_list:
        orbit_file = os.path.join(orbit_folder, "VerticalSectorFFA-trackOrbit.dat")
        if not os.path.exists(orbit_file):
            orbit_file = os.path.join(orbit_folder, "VerticalSectorFFA-trackOrbit_1.dat")
        orbit = plot.LoadOrbit(orbit_file, allowed_events, test_function)
        orbit_list.append(plot.PlotOrbit(orbit, name, None))
    plot_fields = plot.PlotFields(plot.GetFields(lattice_file), orbit_list)
    plot_fields.polygon_plotter = plot.PlotPolygon(10, 2.8)
    plot_fields.log_plotter = log_file
    plot_fields.n_2d_points = 100
    plot_fields.e0 = e0
    job_name = "baseline"
    figure = plot_fields.rf_fig_2("baseline", [r0*math.cos(math.pi*1.0), r0*math.sin(math.pi*1.0), 0.0, t0], [-4.5, -4.2], [-0.5, 0.5])
    figure.savefig(os.path.join(base_dir, job_name+"_rf.png"))



def main():
    #plot_beam()
    plot_azimuthal()
    #plot_rf()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("Press <CR> to finish")

