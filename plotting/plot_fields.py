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
#import pyopal.objects.parser
#import pyopal.objects.track_run

def pybase():
    base_dir = "output/2023-03-01_baseline/find_bump_v14/"
    run_dir = "plotter"
    name = "fields_base"
    orbit_folder_list = glob.glob(os.path.join(base_dir, "tmp/find_closed_orbits"))
    plot_range = [0, 180]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "fets_ffa"

def pybump():
    by = "0.2"
    bumpp = "0.0"
    dx = "-50.0" # only for the field plots
    base_dir = "/home/cr67/work/2017-07-07_isis2/horizontal_isis3/output/2023-03-01_baseline/find_bump_v17/"
    orbit_dir_glob = f"{base_dir}/bump={dx}_by={by}*bumpp=*/"
    run_dir = f"{base_dir}/bump=-50.0_by={by}_bumpp={bumpp}/tmp/find_bump"
    name = f"fields_bump={dx}_by={by}"
    orbit_folder_list = sorted(glob.glob(os.path.join(orbit_dir_glob, "tmp/find_bump")))
    plot_range = [45, 135]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "fets_ffa"

def pytrack():
    by = "0.1"
    bumpp = "-0.1"
    dx = "-50.0" # only for the field plots
    base_dir = "/home/cr67/work/2017-07-07_isis2/horizontal_isis3/output/2023-03-01_baseline/find_bump_v17/"
    orbit_dir_glob = f"{base_dir}/bump=*_by={by}*bumpp={bumpp}/"
    injection_dir = f"{base_dir}/bump={dx}_by={by}*bumpp={bumpp}/"
    run_dir = f"{base_dir}/bump={dx}_by={by}_bumpp={bumpp}/track_beam/reference"
    name = f"fields_track_by={by}"
    orbit_folder_list = sorted(glob.glob(os.path.join(orbit_dir_glob, "track_beam/reference")))
    orbit_folder_list += [os.path.join(injection_dir, "track_beam/injection")]
    plot_range = [45, 135]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "fets_ffa"

def injection():
    by = "0.1"
    bumpp = "-0.05"
    dx1 = "-30.0"
    dx2 = "-20.0"

    base_dir = "/home/cr67/work/2017-07-07_isis2/horizontal_isis3/output/2023-03-01_baseline/find_bump_v17/"
    run_dir = f"{base_dir}/bump={dx1}_by={by}_bumpp={bumpp}/track_bump_full/plotting"

    orbit_folder_list = [
        f"{base_dir}/bump=-0.0_by=0.0_bumpp=0.0/tmp/find_closed_orbits",
        f"{base_dir}/bump={dx2}_by={by}_bumpp={bumpp}/track_beam/reference",
        f"{base_dir}/bump={dx1}_by={by}_bumpp={bumpp}/track_beam/reference",
        f"{base_dir}/bump={dx1}_by={by}_bumpp={bumpp}/track_bump_full/injection",
    ]
    name = f"injection_zoom_out_track_by={by}"

    plot_range = [45, 135]
    return base_dir, run_dir, orbit_folder_list, plot_range, name, "fets_ffa"

def generate_name(folder):
    fname = folder.split("/")[-2]
    fname = fname.replace("_", " ")
    return fname

def hardcoded_contours():
    df = 16/2/math.pi
    phi_cell_list = [11.0472, 20.0472,  32.8310, 41.8310,  48.1690, 57.1690,  69.9528, 78.9528]
    phi0_list = []
    for i in range(4):
        phi0_list += [phi+i*90 for phi in phi_cell_list]
    angle = 30
    r0 = 3.6
    contour_list = [{
            "phi0":phi, "r0":r0, "spiral_angle":angle, "linestyle":"--", "colour":"grey", "label":""
        } for phi in phi0_list]
    phi_mag_list = []
    mag_length = 10+2.13*2+0.05*2/(2*math.pi*3.6)*360
    print("MAG LENGTH", mag_length, "degree")
    for i, phi0 in enumerate(phi0_list[::2]):
        phi_mean = (phi0_list[2*i]+phi0_list[2*i+1])/2
        print(phi_mean)
        cont = {
            "phi0":phi_mean-mag_length/2, "phi1":phi_mean+mag_length/2, "r0":r0, "spiral_angle":angle, "linestyle":"--", "colour":"grey", "label":"", "alpha":0.1,
        }
        contour_list.append(cont)
        contour_list.append(cont)

    return contour_list

def plot_azimuthal():
    base_dir, run_dir, orbit_folder_list, plot_range, job_name, lattice_name = injection()
    r0 = 3.6 # I think this is just used for calculating s
    orbit_folder_list = [(o, "") for o in orbit_folder_list]
    probe_files = "*ring*probe*5.h5" #"FOILPROBE_1.h5" #
    track_files = "*-trackOrbit*dat"
    lattice_file = os.path.join(base_dir, run_dir+"/run_"+lattice_name+".py")
    angle_domain = [0.0, 360.0]
    allowed_events = None # ["ID1"]
    disallowed_events = ["ID0"]
    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    test_function = lambda words: words[1]**2+words[3]**2 > (4.0)**2 or words[1]**2+words[3]**2 < (3.2)**2
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
        orbit = plot.LoadOrbit(orbit_file, allowed_events, disallowed_events, test_function)
        orbit_list.append(plot.PlotOrbit(orbit, name, h5))
        orbit_list[-1].r0 = r0
        print("Loaded", orbit_file)
    if not len(orbit_list):
        raise RuntimeError("Could not find any orbit files", orbit_folder_list)
    print("BASE_DIR", base_dir, "LATTICE FILE", lattice_file)
    plot_fields = plot.PlotFields(plot.GetFields(lattice_file), orbit_list)
    plot_fields.spiral_contours = hardcoded_contours()
    plot_fields.azimuthal_field_plot = ["bz", "div_b", "curl_b"]
    figure = plot_fields.azimuthal_fig("bump", plot_range, do_field_plot = True)
    for axes in figure.axes:
        if "$r_0\\phi$" in axes.get_xlabel():
            continue
        axes.grid(True)
    figname = os.path.join(base_dir, job_name+".png")
    figure.savefig(figname)
    print("Figure saved as", figname)

def plot_beam():
    base_dir, run_dir, orbit_folder_list, plot_range, job_name, lattice_name = pybump()
    rmax = 5.0
    b0 = 1.0
    orbit_folder_list = [(orbit, generate_name(orbit)) for orbit in orbit_folder_list]
    probe_files = "*probe*.h5" #"FOILPROBE_1.h5" #
    lattice_file = os.path.join(base_dir, run_dir+"/run_"+lattice_name+".py")
    log_file = os.path.join(base_dir, run_dir+"log")
    angle_domain = [-0.0, 360.0]
    allowed_events = None #["ID1"]
    disallowed_events = ["ID0"]
    plot.LoadOrbit.azimuthal_domain = angle_domain
    plot.LoadH5.azimuthal_domain = angle_domain
    test_function = lambda words: words[1]**2+words[3]**2 > (rmax)**2
    orbit_list = []
    #log_file = plot.LoadLog(log_file)
    #log_file.element_lambda = lambda element: "MAGNET" in element["name"] and "BUMP" in element["name"]
    #log_file.print()
    for orbit_folder, name in orbit_folder_list:
        h5 = plot.LoadH5(os.path.join(orbit_folder, probe_files))
        orbit_file = glob.glob(os.path.join(orbit_folder, lattice_name+"-trackOrbit.dat"))[0]
        if not os.path.exists(orbit_file):
            orbit_file = os.path.join(orbit_folder, lattice_name+"-trackOrbit_1.dat")
        orbit = plot.LoadOrbit(orbit_file, allowed_events, disallowed_events, test_function)
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
    #plot_beam()
    plot_azimuthal()
    #plot_rf()

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    input("Press <CR> to finish")

