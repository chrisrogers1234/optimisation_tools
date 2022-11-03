"""
Plot a single closed orbit (once tracking has finished)
"""
import glob

import matplotlib
import matplotlib.pyplot

import optimisation_tools.utils.utilities
import optimisation_tools.plotting.plot as plot

def bump_plot_dirs():
    root_dir = "output/arctan_baseline/bump_quest_v9/"
    beam_dir_1 = root_dir+\
               "find_bump_r*0_*/tmp/find_bump/VerticalSectorFFA-trackOrbit*"
    beam_dir_2 = root_dir+\
               "find_bump_r*5_*/tmp/find_bump/VerticalSectorFFA-trackOrbit*"
    dir_list = sorted(glob.glob(beam_dir_1)+glob.glob(beam_dir_2))
    plot_dir = root_dir+"/closed_orbit_plot"
    return dir_list, plot_dir

def baseline_plot_dirs():
    root_dir = "output/arctan_baseline/baseline_test_3/"
    beam_dir = root_dir+\
                "tmp/find_closed_orbit*/VerticalSectorFFA-trackOrbit_1.dat"
    plot_dir = root_dir+"/closed_orbit_plot"
    dir_list = glob.glob(beam_dir)
    print(beam_dir)
    return dir_list, plot_dir

def make_plot(dir_list, plot_dir):
    plot.LoadOrbit.azimuthal_domain = [0, 360]
    r_figure = matplotlib.pyplot.figure()
    r_axes = r_figure.add_subplot(1, 1, 1)
    z_figure = matplotlib.pyplot.figure()
    z_axes = z_figure.add_subplot(1, 1, 1)
    for a_dir in dir_list:
        orbit = plot.LoadOrbit(a_dir, "ID1", lambda words: words[3] < 0)
        plot_orbit = plot.PlotOrbit(orbit)
        plot_orbit.r0 = 4.3
        plot_orbit.plot_2d(r_axes, "phi", "r", [0, 180], [4.225, 4.425])
        plot_orbit.plot_2d(z_axes, "phi", "z", [0, 180], [-0.125, -0.085])
    optimisation_tools.utils.utilities.clear_dir(plot_dir)
    r_figure.savefig(plot_dir+"/orbit_r.png")
    z_figure.savefig(plot_dir+"/orbit_z.png")


def main():
    dir_list, plot_dir = baseline_plot_dirs()
    make_plot(dir_list, plot_dir)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
