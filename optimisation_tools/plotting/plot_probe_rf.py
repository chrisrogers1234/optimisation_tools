import bisect
import json
import os
import math
import glob

import numpy
import matplotlib
import matplotlib.pyplot
import scipy.interpolate
import copy

import xboa.common
import optimisation_tools.plotting.plot
import pyopal.objects.field
import optimisation_tools.utils.polynomial_fitter
import optimisation_tools.utils.decoupled_transfer_matrix

class PlotProbeRF:
    def __init__(self, target_dir = ""):
        self.r0 = 3.6
        self.verbose = 1
        self.plot_dir = target_dir
        self.run_dir = os.path.join(self.plot_dir, "track_beam/plotting/")
        self.track_glob = os.path.join(self.plot_dir, "track_beam/track?")
        self.file_glob = "ring_probe_001.h5"
        self.lattice_file = os.path.join(self.run_dir, "run_fets_ffa.py")
        self.load_field = True
        self.field_loader = None
        self.particle_data = []
        self.station_data = []
        self.frequency = 1.0 # Hz
        self.mass = xboa.common.pdg_pid_to_mass[2212]
        self.ref_track = 0 # used for calculating reference time-energy
        self.events_selection = "time"
        self.plot_station = None
        self.plot_time = None
        self.acceptance_kick_time = 0
        self.color_variable = "station"

    def get_time_offset(self, track_folder):
        mHz_to_Hz = 1e-3
        subs_str = open(os.path.join(track_folder, "subs.json")).read()
        subs_json = json.loads(subs_str)
        cavity_0 = subs_json["__rf_list__"][0]
        phase = cavity_0["phase"][0]
        self.frequency = cavity_0["frequency"][0]*mHz_to_Hz # Gah!
        time_offset = +phase/2/math.pi/self.frequency
        print("Frequency", self.frequency)
        return time_offset

    def load_h5(self):
        self.particle_data = []
        for track_name in glob.glob(self.track_glob):
            file_glob = os.path.join(track_name, self.file_glob)
            h5_loader = optimisation_tools.plotting.plot.LoadH5(file_glob, verbose = 0, will_load = False)
            h5_loader.id_include = [1]
            h5_loader.load_probes()
            h5_loader.data = sorted(h5_loader.data, key = lambda x: x["t"])
            mass = xboa.common.pdg_pid_to_mass[2212]
            print("Loaded", len(h5_loader.data), "tracks from", file_glob)
            time_offset = self.get_time_offset(track_name)
            for hit in h5_loader.data:
                hit["t"] += time_offset
            print("   ... applied time offset ", time_offset)
            self.particle_data.append(h5_loader.data)
        self.ref_spline = self.get_et_spline(self.particle_data[self.ref_track])


    def load_closed_orbit(self):
        in_string = open(os.path.join(self.plot_dir, "closed_orbits_cache")).read()
        self.closed_orbit = json.loads(in_string)
        for hit in self.closed_orbit[-1]["ref_track"]:
            r = (hit["x"]**2+hit["y"]**2)**0.5
            print(f"{hit['station']} {hit['t']} {r}")


    def load_lattice(self):
        subs_str = open(os.path.join(self.run_dir, "subs.json")).read()
        self.subs = json.loads(subs_str)
        if self.load_field:
            self.field_loader = optimisation_tools.plotting.plot.GetFields(self.lattice_file)
            if self.verbose:
                print(self.field_loader.str_field())

    def plots(self):
        self.plot_lattice()
        self.plot_rf_radial()
        self.plot_rf_azimuthal()
        self.plot_rf_waveform([3.6, 3.7, 3.8], [90, 270], [0.0, 100e-6])

    def station_plots(self):
        if self.events_selection == "time":
            self.events_by_time()
        elif self.events_selection == "station":
            self.events_by_station()
        elif self.events_selection == "all":
            self.events_all()
        else:
            raise KeyError(f"Events selection not recognised: {self.events_selection}")

        self.plot_energy()
        self.plot_energy_dt()

    def get_et_spline(self, track_beam_run):
        t_list = []
        e_list = []
        for item in track_beam_run:
            if item["id"] != 1:
                continue
            if item["station"] % 16 != 0:
                continue
            energy = (item["px"]**2+item["py"]**2+item["pz"]**2+self.mass**2)**0.5-self.mass
            if len(e_list) == 0 or energy > e_list[-1]:
                e_list.append(energy)
                t_list.append(item["t"])
                if energy < 3.2:
                    print("get_et_spline   ", t_list[-1], e_list[-1])
        ref_spline = scipy.interpolate.CubicSpline(e_list, t_list, extrapolate = True)
        return ref_spline

    def get_dt(self, item, energy):
        dt = item["t"] - self.ref_spline(energy)
        dt += 0.5/self.frequency
        dt -= int(dt*self.frequency)/self.frequency
        dt -= 0.5/self.frequency
        if energy < 3.2 and False:
            print("get_dt", energy, dt, item["t"], self.ref_spline(energy))
        return dt

    def events_all(self):
        self.station_data = []
        for i, track_beam_run in enumerate(self.particle_data):
            for item in track_beam_run:
                self.station_data.append(item)

    def events_by_station(self):
        self.station_data = []
        for i, track_beam_run in enumerate(self.particle_data):
            for item in track_beam_run:
                if item["station"] == self.plot_station:
                    self.station_data.append(item)

    def events_by_time(self):
        self.station_data = []
        for i, track_beam_run in enumerate(self.particle_data):
            data = []
            for item in track_beam_run:
                if item["t"] > self.plot_time:
                    data.append(item)
            data = sorted(data, key=lambda x: x["t"])
            if len(data):
                self.station_data.append(data[0])

    def acceptance(self, event_id):
        return 0

    def station_title(self, is_fname = False):
        if self.events_selection == "time":
            if is_fname:
                return f"time_{self.plot_time:.4g}"
            return f"Time {self.plot_time*1e9:.4g} ns"
        elif self.events_selection == "station":
            if is_fname:
                return f"station_{self.plot_station:04}"
            return f"Station {self.plot_station}"
        elif self.events_selection == "all":
            if is_fname:
                return "all_stations"
            return "All Stations"

    def plot_energy_dt(self):
        s_to_ns = 1e9
        dt_figure = matplotlib.pyplot.figure(num="dt vs KE")
        dt_figure.clf()
        dt_axes = dt_figure.add_subplot(1, 1, 1)

        dt_list = []
        e_list = []
        color_list = []
        for hit in self.station_data:
            energy = hit["energy"] #(item["px"]**2+item["py"]**2+item["pz"]**2+self.mass**2)**0.5-self.mass
            dt = self.get_dt(hit, energy)
            dt_list.append(dt*self.frequency*360)
            e_list.append(energy)
            color_list.append(hit[self.color_variable])
        dt_axes.scatter(dt_list, e_list, s=1, c=color_list)
        dt_axes.set_title(self.station_title())
        dt_axes.set_xlabel("RF phase [degree]")
        dt_axes.set_ylabel("E [MeV]")
        #dt_axes.legend()
        dt_figure.savefig(os.path.join(self.plot_dir, f"dt_vs_e_{self.station_title(True)}.png"))

    def plot_energy(self):
        s_to_ns = 1e9
        figure = matplotlib.pyplot.figure(num="t vs KE")
        figure.clf()
        axes = figure.add_subplot(1, 1, 1)
        r_figure = matplotlib.pyplot.figure(num="r vs KE")
        r_figure.clf()
        r_axes = r_figure.add_subplot(1, 1, 1)

        r_list = []
        er_list = []
        t_list = []
        e_list = []
        color_list = []
        for hit in self.station_data:
            energy = hit["energy"]
            e_list.append(energy)
            t_list.append(hit["t"]*s_to_ns)
            r_list.append(hit["r"])
            er_list.append(energy)
            color_list.append(hit[self.color_variable])
        axes.scatter(t_list, e_list, s=1, c=color_list)
        axes.set_xlabel("t [ns]")
        axes.set_ylabel("E [MeV]")
        axes.set_title(self.station_title())
        #axes.legend()
        figure.savefig(os.path.join(self.plot_dir, f"t_vs_e_{self.station_title(True)}.png"))

        r_axes.scatter(r_list, er_list, s=1, c=color_list)
        r_axes.set_xlabel("r [m]")
        r_axes.set_ylabel("E [MeV]")
        r_axes.set_title(self.station_title())
        #r_axes.legend()
        r_figure.savefig(os.path.join(self.plot_dir, f"r_vs_e_{self.station_title(True)}.png"))

    def plot_lattice(self):
        if not self.load_field:
            return
        bz_list = []
        ephi_list = []
        theta_list = []
        for i in range(3600):
            angle_rad = math.radians(i/10.0)
            theta_list.append(i/10.0)
            x = self.r0*math.sin(angle_rad)
            y = self.r0*math.cos(angle_rad)
            field = pyopal.objects.field.get_field_value(x, y, 0.0, 1e6)
            bz_list.append(field[3])
        ephi_list = [self.get_ephi(self.r0, theta_deg, 0) for theta_deg in theta_list]
        figure = matplotlib.pyplot.figure(num="B vs theta")
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(theta_list, bz_list)
        axes.set_xlabel("$\\theta$ [degree]")
        axes.set_ylabel("B$_{z}$ [T]")
        figure.savefig(os.path.join(self.plot_dir, "theta_vs_By.png"))

        figure = matplotlib.pyplot.figure(num="Ephi vs theta")
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(theta_list, ephi_list)
        axes.set_ylabel("E$_{\\phi}$ [MV/m]")
        figure.savefig(os.path.join(self.plot_dir, "theta_vs_Ephi.png"))

    def find_tm(self, target_run, station_0, station_1):
        data = self.particle_data[target_run]
        data_0 = []
        data_1 = []
        variables = ["r", "pr", "t", "energy"]
        units = {"r":1.0, "pr":1.0, "t":1e9, "energy":1.0}
        for item in data:
            if item["station"] == station_0:
                data_0.append(item)
            if item["station"] == station_1:
                data_1.append(item)
        data_0 = sorted(data_0, key=lambda x: x["id"])
        data_1 = sorted(data_1, key=lambda x: x["id"])
        for i in range(max(len(data_0), len(data_1))):
            while data_0[i]["id"] > data_1[i]["id"]:
                del data_1[i]
            while data_1[i]["id"] > data_0[i]["id"]:
                del data_0[i]
            assert data_0[i]["id"] == data_1[i]["id"]
        ref_0 = copy.deepcopy(data_0[0])
        data_0 = [[(item[var]-ref_0[var])*units[var] for var in variables] for item in data_0]
        ref_1 = copy.deepcopy(data_1[0])
        data_1 = [[(item[var]-ref_1[var])*units[var] for var in variables] for item in data_1]
        fitter = optimisation_tools.utils.polynomial_fitter.PolynomialFitter(len(variables))
        tm = fitter.fit_transfer_map(data_0, data_1)
        print("Using data 0 for station", station_0)
        for row in data_0:
            for value in row:
                print(f"{value:14.6g} ", end="")
            print()
        print("Using data 1 for station", station_1)
        for row in data_1:
            for value in row:
                print(f"{value:14.6g} ", end="")
            print()
        print("Found TM for station", station_0, "to", station_1, "from", len(data_0), "hits")
        self.print_tm(tm)
        matrix = numpy.array([row[1:] for row in tm])
        print("Symplecticity:")
        simple = optimisation_tools.utils.decoupled_transfer_matrix.DecoupledTransferMatrix.simplecticity(matrix)
        print(simple)
        print("Determinant:", numpy.linalg.det(matrix), "\n\n")


    def print_tm(self, tm):
        for i, row in enumerate(tm):
            sym = " "
            if i == int(len(tm)/2):
                sym = "+"
            print(f"{row[0]:12.4g} {sym} (", end="")
            for value in row[1:]:
                print(f"{value:12.4g} ", end="")
            print(f")")

    def get_ephi(self, r, phi_deg, t):
        VtoMV = 1e-6
        phi_rad = math.radians(phi_deg)
        x = r*math.cos(phi_rad)
        y = r*math.sin(phi_rad)
        field = pyopal.objects.field.get_field_value(x, y, 0.0, t)
        ephi = field[5]*math.cos(phi_rad) - field[4]*math.sin(phi_rad)
        ephi = ephi*VtoMV
        return ephi # that is in MV/m

    def plot_rf_radial(self):
        if not self.load_field:
            return
        figure, axes = self.new_axes("r vs Ephi")
        for phi_deg, phi_rf in [(90, 90), (270, 0)]:
            t = 1e-4*phi_rf/360.0
            r_list = []
            e_list = []
            for ri in range(500+1):
                r = ri*0.001+3.5
                r_list.append(r)
                e_list.append(self.get_ephi(r, phi_deg, t))
            axes.plot(r_list, e_list, label=f"$\\phi$={phi_deg}$^\\circ$")
        axes.set_xlabel("r [m]")
        axes.set_ylabel("E$_y$ [MV/m]")
        axes.legend()
        figure.savefig(os.path.join(self.plot_dir, "r_vs_ey.png"))

    def get_ring_tof(self, r):
        return 0.0

    def plot_rf_waveform(self, r_list, phi_list, t_range):
        s_to_ns = 1e9
        t_list = numpy.linspace(t_range[0], t_range[1], 100)
        t_plot_list = [t*s_to_ns for t in t_list]
        for r in r_list:
            figure, axes = self.new_axes(f"t vs Ephi r={r}")
            etot_list = [0.0 for t in t_list]
            axes.set_title(f"r={r} m")
            for phi in phi_list:
                ephi_list = [self.get_ephi(r, phi, t) for t in t_list]
                etot_list = [etot+ephi_list[i] for i, etot in enumerate(etot_list)]
                axes.plot(t_plot_list, ephi_list, label=f"$\\phi={phi}^\\circ$")
            axes.plot(t_plot_list, etot_list, color="black", linestyle="dashed")
            axes.set_xlabel("t [ns]")
            axes.set_ylabel("E$_{\\phi}$ [MV/m]")
            axes.legend()
            figure.savefig(os.path.join(self.plot_dir, f"t_vs_Ephi_r_{r}.png"))

    def plot_rf_azimuthal(self):
        if not self.load_field:
            return

        for ti in range(0, 1001, 10):
            t = ti*1000*1e-9
            r = 3.6
            print_str = f"{t:8.4g} {r:8.4g}"
            for phi_deg in [0.0, 90.0, 180, 270]:
                print_str += f" phi={phi_deg}: {self.get_ephi(r, phi_deg, t)}"
            #print(print_str)

    def new_axes(self, window_title):
        while matplotlib.pyplot.fignum_exists(window_title):
            window_title = window_title+"_"
        figure = matplotlib.pyplot.figure(num=window_title)
        axes = figure.add_subplot(1, 1, 1)
        return figure, axes

def ring_tof(radius):
    r0 = 3.6
    p0 = 75.091
    k = 7.4561
    circumference_factor = 996.2906931/1057.3312719268977
    momentum = p0*(radius/r0)**(k+1)
    mass = xboa.common.pdg_pid_to_mass[2212]
    energy = (momentum**2+mass**2)**0.5
    velocity = momentum/energy*xboa.common.constants["c_light"]*1e6 # m/s
    tof = radius*2*math.pi*circumference_factor/velocity # seconds
    print("radius", radius, "momentum", momentum, "velocity", velocity, "tof", tof)
    return tof

def main():
    for run_dir in glob.glob("output/2023-03-01_baseline/rf_stuff_2/v10"):
        rf_plotter = PlotProbeRF(run_dir)
        rf_plotter.track_glob = os.path.join(rf_plotter.plot_dir, "track_beam/track*")
        rf_plotter.get_ring_tof = ring_tof
        rf_plotter.file_glob = "ring_probe_001.h5"
        rf_plotter.load_h5()
        rf_plotter.load_lattice()
        #rf_plotter.load_closed_orbit()
        rf_plotter.plots()
        for station in []: #[i for i in range(6)]+[i for i in range(10, 500, 10)]:
            print("\rPlotting station", station, end="")
            rf_plotter.events_selection = "all" # "time" "station" "all"
            rf_plotter.plot_station = station
            rf_plotter.plot_time = station*1000*1e-9 # seconds
            rf_plotter.station_plots()
        rf_plotter.events_selection = "all" # "time" "station" "all"
        rf_plotter.station_plots()
        print()
        return
        rf_plotter.find_tm(1, 0, 0)
        rf_plotter.find_tm(1, 0, 16)
        rf_plotter.find_tm(1, 0, 32)
        rf_plotter.find_tm(1, 16, 32)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish\n")