import json
import math
import os

import matplotlib
import matplotlib.pyplot

import xboa.hit
import plotting.plot as plot

def field_plot(axes, ref_orbit, fields):
    phi_step = 0.1
    z_step = -0.01
    phi_list = [phii*phi_step for phii in range(72)]
    r_list = ref_orbit.interpolate("phi", "r", phi_list)[0]
    z_list = sorted([z*z_step for z in range(100)])
    z_hlist = []
    phi_hlist = []
    bz_hlist = []
    for phii, phi in enumerate(phi_list):
        for z in z_list:
            x = r_list[phii]*math.cos(math.radians(phi))
            y = r_list[phii]*math.sin(math.radians(phi))
            z_hlist.append(z)
            phi_hlist.append(phi)
            bz_hlist.append(fields.get_field(x, y, z, 0.0)["bz"])
    z_bins = [z-z_step/2.0 for z in z_list]
    z_bins.append(z_bins[-1]+abs(z_step))
    phi_bins = [phi-phi_step/2.0 for phi in phi_list]
    phi_bins.append(phi_bins[-1]+abs(phi_step))
    axes.hist2d(phi_hlist, z_hlist, [phi_bins, z_bins], weights=bz_hlist)
    print("plotted fields")
    

def plot_azimuthal():
    base_dir = "output/double_triplet_baseline/energy_scaling/3.0/"
    track_file = base_dir+"tmp/find_closed_orbits/VerticalSectorFFA-trackOrbit_1.dat"
    lattice_file = os.path.join(base_dir, "tmp/find_closed_orbits/VerticalSectorFFA.tmp")
    angle_domain = [0.0, 360.0] #[104, 112] #  
    allowed_events = ["ID1"]

    plot.LoadOrbit.azimuthal_domain = angle_domain
    test_function = lambda words: words[5] > 0.1
    fields = plot.GetFields(lattice_file)

    orbit = plot.LoadOrbit(track_file, allowed_events, test_function)
    plotter = plot.PlotOrbit(orbit, "3.0 MeV", None)
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    field_plot(axes, orbit, fields)
    #plotter.plot_2d(axes, "phi", "z", [0.0, 72.0], [-1.0, 0.0])
    figure.savefig(base_dir+"azimuthal.png")

def plot_my_da(plot_dir, da_list, axis_label):
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    var_x = axis_label
    var_y = axis_label+"'"
    for da_seed in da_list:
        seed = da_seed[0]
        hit_list = da_seed[1]
        hit_list = [xboa.hit.Hit.new_from_dict(hit) for hit in hit_list]
        x = [hit[var_x] for hit in hit_list]
        xp = [hit[var_y] for hit in hit_list]
        axes.scatter(x, xp, s=1)
    axes.set_xlabel(plot.Labels.labels[var_x])
    axes.set_ylabel(plot.Labels.labels[var_y])
    figure.savefig(plot_dir + "da_"+var_x+".png")


def plot_da():
    base_dir = "output/double_triplet_baseline/energy_scaling/3.0/"
    fin = open(base_dir+"get_da.tmp")
    da_json = json.loads(fin.readline())
    plot_my_da(base_dir, da_json["x_da"], "x")
    plot_my_da(base_dir, da_json["y_da"], "y")
 
def main():
    plot_da()

if __name__ == "__main__":
    main()
