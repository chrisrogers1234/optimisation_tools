"""
Plot a single closed orbit (once tracking has finished)
"""
import time
import sys
import copy
import os
import math
import argparse
import h5py
import glob
import shutil
import json
import bisect
import numpy
import math
import importlib

import itertools

import matplotlib
import ROOT

import xboa.common

import pyopal.objects.parser
import pyopal.objects.field
import pyopal.objects.ffa_field_mapper
import optimisation_tools.utils.utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.twod_transfer_matrix import TwoDTransferMatrix
DecoupledTransferMatrix.det_tolerance = 1e-2

class GetFields(object):
    def __init__(self, lattice_file):
        self.dx = 0.001
        self.dt = 0.001
        self.lattice_file = lattice_file
        if self.lattice_file[-3:] == ".py":
            print("Detected python script, loading")
            self.python_lattice()
        else:
            print("Assume this is an opal lattice, loading")
            self.hack_lattice()
            self.load_lattice()
        self.subs = {}

    def hack_lattice(self):
        source = self.lattice_file
        src_dir = "/".join(self.lattice_file.split("/")[:-1])
        target_dir = src_dir+"/hack/"
        optimisation_tools.utils.utilities.clear_dir(target_dir)
        shutil.copy2(src_dir+"/disttest.dat", target_dir+"/disttest.dat")
        prefix = self.lattice_file.split(".")[:-1]+["hack"]
        target = ".".join(prefix)
        target = self.lattice_file.split("/")
        target.insert(-1, "hack")
        target = "/".join(target)
        print("Hacking from\n   ", source, "to\n   ", target)
        if source == target:
            raise RuntimeError("Source == target! oh noes!")
        fin = open(source, "r")
        fout = open(target, "w")
        for line in fin.readlines():
            if "ASCIIDUMP=" in line:
                line = "Option, ASCIIDUMP=True; // plotfields HACK!!! \n"
            if "ENABLEHDF5=" in line:
                line = "Option, ENABLEHDF5=False; // plotfields HACK!!! \n"
            if "REAL N_TURNS=" in line:
                line = "REAL N_TURNS=0.001; // plotfields HACK!!! \n"
            if "BOOL DO_MAGNET_FIELD_MAPS=" in line:
                line = "BOOL DO_MAGNET_FIELD_MAPS=False; // plotfields HACK!!! \n"
            fout.write(line)
        self.lattice_file = target
        fin.close()

    def python_lattice(self):
        script_path, script_file_name = os.path.split(self.lattice_file)
        script_module = script_file_name.replace(".py", "")
        sys.path.append(script_path)
        try:
            here = os.getcwd()
            script_mod = importlib.import_module(script_module)
            importlib.reload(script_mod)
            os.chdir(script_path)
            script_mod.build_field()
        except ModuleNotFoundError:
            print("Failed to find python module", script_module, "in", os.getcwd(), "using", script_path)
        finally:
            sys.path.remove(script_path)
            os.chdir(here)

    def load_lattice(self):
        here = os.getcwd()
        a_dir, a_file = os.path.split(self.lattice_file)
        os.chdir(a_dir)
        pyopal.objects.parser.initialise_from_opal_file(a_file)
        os.chdir(here)

    def get_derivative(self, var1, var2, x, y, z, t):
        pos_vec = [x, y, z, t]
        var2 = ["x", "y", "z", "t"].index(var2)
        pos_vec[var2] += self.dx
        field_plus = self.get_field(*pos_vec)[var1]
        pos_vec[var2] -= 2*self.dx
        field_minus = self.get_field(*pos_vec)[var1]
        derivative = (field_plus-field_minus)/2.0/self.dx
        return derivative

    def get_div_b(self, x, y, z, t):
        div_b = self.get_derivative("bx", "x", x, y, z, t) + \
                self.get_derivative("by", "y", x, y, z, t) + \
                self.get_derivative("bz", "z", x, y, z, t)
        return div_b

    def get_curl_b(self, x, y, z, t):
        curl_b = [
            self.get_derivative("by", "z", x, y, z, t) - \
            self.get_derivative("bz", "y", x, y, z, t),
            self.get_derivative("bx", "z", x, y, z, t) - \
            self.get_derivative("bz", "x", x, y, z, t),
            self.get_derivative("bx", "y", x, y, z, t) - \
            self.get_derivative("by", "x", x, y, z, t)
        ]
        return curl_b

    def get(self, var, x, y, z, t):
        if var in self.field_vars:
            return self.get_field(x, y, z, t)[var]
        elif var == "div_b":
            return self.get_div_b(x, y, z, t)
        elif var == "curl_b":
            curl_b = self.get_curl_b(x, y, z, t)
            curl_b = (curl_b[0]**2+curl_b[1]**2+curl_b[2]**2)**0.5
            return curl_b

    def get_field(self, x, y, z, t):
        phi = math.atan2(y, x)
        oob, bx, by, bz, ex, ey, ez = \
                                pyopal.objects.field.get_field_value(x, y, z, t)
        btot = (bx**2+by**2+bz**2)**0.5
        br = bx*math.cos(phi)+by*math.sin(phi)
        bphi =  -bx*math.sin(phi)+by*math.cos(phi)

        etot = (ex**2+ey**2+ez**2)**0.5
        er = ex*math.cos(phi)+ey*math.sin(phi)
        ephi =  -ex*math.sin(phi)+ey*math.cos(phi)
        field = {"bx":bx, "by":by, "bz":bz, "btot":btot, "br":br, "bphi":bphi,
                 "ex":ex, "ey":ey, "ez":ez, "etot":etot, "er":er, "ephi":ephi,
                 "oob":oob}
        return field

    def str_field(self):
        my_string = ""
        n_fields = pyopal.objects.field.get_number_of_elements()
        for i in range(n_fields):
            name = pyopal.objects.field.get_element_name(i).ljust(20)
            start_pos = [f"{pos:.5g}" for pos in pyopal.objects.field.get_element_start_position(i)]
            my_string += f"{name} {start_pos}\n"
        return my_string

    field_vars = ["bx", "by", "bz", "btot", "br", "bphi", "ex", "ey", "ez", "etot", "er", "ephi", "oob"]

class LoadLog(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.logged_elements = []
        self.fmt_str = "8.5g"
        self.element_lambda = None
        self.load_log()

    def load_log(self):
        log_file = open(self.file_name)
        line = "A"
        while line != "":
            line = log_file.readline()
            if "Added " not in line:
                continue
            if " to Ring" not in line:
                continue
            element = {}
            name = line.split("Added ")[1].split(" to Ring")[0]
            element["name"] = name
            line = log_file.readline()
            words = line.split("(")
            element["entrance_pos"] = self.scrape_vector("("+words[1], units=1e-3)
            element["entrance_normal"] = self.scrape_vector("("+words[2])
            element["entrance_vertical"] = self.scrape_vector("("+words[3])
            line = log_file.readline()
            words = line.split("(")
            element["exit_pos"] = self.scrape_vector("("+words[1], units=1e-3)
            element["exit_normal"] = self.scrape_vector("("+words[2])
            element["exit_vertical"] = self.scrape_vector("("+words[3])
            element["centre_pos"] = [0.0, 0.0, 0.0]
            for i in range(3):
                element["centre_pos"][i] = (element["entrance_pos"][i]+element["exit_pos"][i])/2.0
            self.logged_elements.append(element)
        print("Loaded", len(self.logged_elements), " elements from log file", self.file_name)

    def scrape_vector(self, line, units=1.0):
        vector = line.split("(")[1]
        vector = vector.split(")")[0]
        vector = [float(number)*units for number in vector.split(",")]
        return vector

    def print(self):
        for element in self.logged_elements:
            if self.element_lambda != None and not self.element_lambda(element):
                continue
            centre = element["centre_pos"]
            r = (centre[0]**2+centre[1]**2)**0.5
            phi = math.degrees(math.atan2(centre[1], centre[0]))
            print(element["name"],
                  "at r:", format(r, self.fmt_str),
                  "phi:", format(phi, self.fmt_str),
                  "x:", format(centre[0], self.fmt_str),
                  "y:", format(centre[1], self.fmt_str))

    def plot(self, axes):
        x_list = []
        y_list = []
        for element in self.logged_elements:
            if self.element_lambda != None and not self.element_lambda(element):
                continue
            x_list.append(element["centre_pos"][0])
            y_list.append(element["centre_pos"][1])
        axes.scatter(x_list, y_list, label="elements")



class LoadOrbit(object):
    def __init__(self, file_name, allowed_id, disallowed_id=[], test_function = None, verbose = 1):
        self.file_name = file_name
        self.orbit = {}
        self.allowed_id = allowed_id
        self.disallowed_id = disallowed_id
        self.test_function = test_function
        self.verbose = verbose
        self.parse_file()
        self.r_phi_track_file()

    def parse_file(self):
        heading = self.heading
        fin = open(self.file_name)
        for item in heading:
            self.orbit[item] = []
        line = fin.readline().rstrip("\n")
        line_count = 0
        while line != "":
            line = fin.readline()
            line_count += 1
            words = line.rstrip("\n").split()
            if len(words) != len(heading):
                if self.verbose:
                    print("Line\n  "+line+"\nmismatched to heading\n  "+str(heading)+"\nin parse_file "+self.file_name)
            else:
                words = [self.types[i](x)*self.units[i] for i, x in enumerate(words)]
                if self.allowed_id and words[0] not in self.allowed_id:
                    continue
                if self.disallowed_id and words[0] in self.disallowed_id:
                    continue
                is_okay = self.test_function == None or not self.test_function(words)
                if not is_okay:
                    if self.verbose:
                        print("Stopping due to failed test function at", words, self.test_function(words))
                    break
                for i, item in enumerate(heading):
                    self.orbit[item].append(words[i])
        if self.verbose > 10:
            print("Got", len(self.orbit["x"]), "/", line_count,
                  "lines from file "+self.file_name,
                  "with allowed_id:", self.allowed_id)

    @classmethod
    def fix_domain(self, phi):
        if phi < self.azimuthal_domain[0]:
            phi += (self.azimuthal_domain[1]-self.azimuthal_domain[0])
        elif phi >  self.azimuthal_domain[1]:
            phi -= (self.azimuthal_domain[1]-self.azimuthal_domain[0])
        return phi

    def r_phi_track_file(self):
        data = self.orbit
        data["r"] = list(range(len(data["x"])))
        data["phi"] = list(range(len(data["x"])))
        data["pr"] = list(range(len(data["x"])))
        data["pphi"] = list(range(len(data["x"])))
        for i in range(len(data["r"])):
            data["r"][i] = (data["x"][i]**2+data["y"][i]**2.)**0.5
            phi = math.atan2(data["y"][i], data["x"][i])
            data["phi"][i] = self.fix_domain(math.degrees(phi))
            px = data["px"][i]
            py = data["py"][i]
            data["pr"][i]   = px*math.cos(phi)+py*math.sin(phi)
            data["pphi"][i] = -px*math.sin(phi)+py*math.cos(phi)

    def interpolate(self, var_x, var_y, x_list):
        """
        var_y can be a single value or a list of values
        Assumes var_x is single valued, and lengths are consistent
        Return value is always [x_list, y_lists_0, y_list_1, ...]
        """
        var_y_list = var_y
        if type(var_y) != type([]):
            var_y_list = [var_y]
        n_var_y = len(var_y_list)
        x_list = sorted(x_list)
        n_orbit_hits = len(self.orbit[var_x]) # length of orbit data
        if n_orbit_hits < 2:
            raise ValueError(str(n_orbit_hits)+" points is not enough to interpolate orbit "+self.file_name)
        my_data = [tuple([self.orbit[var_x][i]]+[self.orbit[var_y][i] for var_y in var_y_list]) for i in range(n_orbit_hits)]
        my_data = sorted(my_data)
        y_list_of_lists = [[None]*len(x_list) for i in range(n_var_y)]
        index1 = bisect.bisect(my_data, tuple([x_list[0]]+[0]*n_var_y) )
        if index1 == 0:
            index1 = 1
            index0 = 0
        else:
            index0 = index1 - 1
        for xi, x in enumerate(x_list):
            while index1+1 < n_orbit_hits and x > my_data[index1][0]:
                index1 += 1
                index0 += 1
            x0 = my_data[index0][0]
            x1 = my_data[index1][0]
            for var_yi in range(n_var_y):
                y0 = my_data[index0][var_yi+1]
                y1 = my_data[index1][var_yi+1]
                y = (y1-y0)/(x1-x0)*(x-x0)+y0
                y_list_of_lists[var_yi][xi] = y
        return [x_list]+y_list_of_lists

    def get_kinetic_energy(self, index):
        px = self.orbit["px"][index]
        py = self.orbit["py"][index]
        pz = self.orbit["pz"][index]
        ke = (px**2+py**2+pz**2+self.p_mass**2)**0.5-self.p_mass
        return ke

    p_mass = xboa.common.pdg_pid_to_mass[2212]
    heading = ["id", "x", "px", "y", "py", "z", "pz"]
    units = [1, 1.0, p_mass, 1.0, p_mass, 1.0, p_mass]
    types = [str]+[float]*6
    azimuthal_domain = [-180.0, 180.0]

class Labels(object):
    labels = {
        "x":"x [m]", "y":"y [m]", "z":"z [m]",
        "r":"r [m]", "phi":"$\\phi$ [$^o$]",
        "px":"p$_{x}$ [MeV/c]", "py":"p$_{y}$ [MeV/c]", "pz":"p$_{z}$ [MeV/c]",
        "pr":"p$_{r}$ [MeV/c]", "pphi":"p$_{\\phi}$ [MeV/c]",
        "bx":"B$_{x}$ [T]", "by":"B$_{y}$ [T]", "bz":"B$_{z}$ [T]",
        "btot":"B$_{tot}$ [T]", "br":"B$_{r}$ [T]", "bphi":"B$_{\\phi}$ [T]",
        "ex":"E$_{x}$ [MV/m]", "ey":"E$_{y}$ [MV/m]", "ez":"E$_{z}$ [MV/m]",
        "etot":"E$_{tot}$ [MV/m]", "er":"E$_{r}$ [MV/m]", "ephi":"E$_{\\phi}$ [MV/m]",
        "div_b":"$\\nabla \\cdot \\mathbf{B}$ [T/m]",
        "curl_b":"$|\\nabla \\times \\mathbf{B}|$ [T/m]",
        "t":"Time [ns]",
        "u":"u", "v":"v", "u'":"u'", "v'":"v'", "pu":"p$_{u}$", "pv":"p$_{v}$",
        "phiu":"$\\phi_{u}$ [rad]", "phiv":"$\\phi_{v}$ [rad]",
        "au":"A$_{u}$ [mm]", "av":"A$_{v}$ [mm]",
        "x'":"x'", "y'":"y'", "z'":"z'", "r'":"r'", "phi'":"$r \\phi'}{ds}'$"
    }

    @classmethod
    def get_colors(cls):
        colors_cycle = itertools.cycle(matplotlib.pyplot.rcParams["axes.prop_cycle"].by_key()["color"])
        colors_list = [(var, next(colors_cycle)) for var in cls.labels.keys()]
        colors_dict = dict(colors_list)
        return colors_dict

class RFAnalyser(object):
    def __init__(self, fields):
        self.fields = fields
        self.n_points = 100

    def get_data(self, centre, var, time_window):
        self.x_list = []

    def get_seed(self):
        pass

    def sine_fit(self, centre, var, time_window):
        voltage = max(self.y_list)
        frequency = 1e-3
        crossings = []
        for i, y in enumerate(self.y_list[1:]):
            if y > 0. and self.y_list[i] < 0: 
                crossings.append(i)
        if len(crossings) > 0:
            t0 = self.x_list[crossings[0]]
        print(crossings)#[1], crossings[0]
        #print "FIT CROSSINGS", self.x_list[crossings[1]], self.x_list[crossings[0]]
        if len(crossings) > 1:
            frequency = 1./(self.x_list[crossings[1]]-self.x_list[crossings[0]])
        frequency *= 2.*math.pi
        print("Seeding sine fit with", t0, frequency, voltage)
        fitter = ROOT.TF1("sin "+str(len(self.root_objects)), "[0]*sin([1]*(x-[2]))")
        fitter.SetParameter(0, voltage)
        fitter.SetParameter(1, frequency)
        fitter.SetParameter(2, t0)
        fitter.SetRange(min(self.x_list), max(self.x_list))
        #fitter.Draw("SAME")
        self.graph.Fit(fitter)
        self.canvas.Update()
        self.root_objects.append(fitter)
        rf_parameters = {
            "voltage":fitter.GetParameter(0),
            "frequency":fitter.GetParameter(1)/2./math.pi,
            "t0":fitter.GetParameter(2)
        }
        return rf_parameters

class PlotFields(object):
    def __init__(self, fields, orbit_plotter = [], polygon_plotter=None):
        self.n_1d_points = 1000
        self.n_2d_points = 1000
        self.fields = fields
        self.orbit_list = orbit_plotter
        self.cmap = "PiYG"
        self.r_min = 3.50
        self.r_max = 4.50
        self.n_phi = 10
        self.b0 = 1.0
        self.e0 = 1.0
        self.t0 = 1000.0
        self.do_pipe = False
        self.polygon_plotter = polygon_plotter
        self.log_plotter = None
        self.spiral_contours = None
        self.azimuthal_field_plot = ['br', 'bphi', 'bz']


    def field_fig(self, job_name, centre, range_x, range_y, do_elevation=False):
        """
        job_name: string name used for figure suptitle
        centre: 
        """
        figure = matplotlib.pyplot.figure(figsize=(20, 10))

        axes = figure.add_subplot(2, 3, 1) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.b0, self.b0], "x", "y", "bx")
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        axes = figure.add_subplot(2, 3, 2)
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.b0, self.b0], "x", "y", "by")
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        axes = figure.add_subplot(2, 3, 3) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.b0, self.b0], "x", "y", "bz")
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        axes = figure.add_subplot(2, 3, 4) 
        if not do_elevation:
            #self.plot_1d(axes, centre, range_y, [-10.0, 10.0], "y", ["bx", "by", "bz", "btot", "div_b", "curl_b"], 0, "")
            self.plot_1d_orbit(axes, self.orbit_list[0], 'phi', ['bx', 'by', 'bz', "curl_b", "div_b"], [0.0, 48.0], [-self.b0, self.b0])
        else:
            for orbit in self.orbit_list:
                orbit.plot_phi_z(axes)

        axes = figure.add_subplot(2, 3, 5) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-0.1, 0.1], "x", "y", "div_b")
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        axes = figure.add_subplot(2, 3, 6) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-0.1, 0.1], "x", "y", "curl_b")
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)
        axes.legend()

        figure.suptitle(job_name)
        return figure

    def one_field_fig(self, job_name, centre, range_x, range_y, field_var):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.b0, self.b0], "x", "y", field_var)
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)
        #axes.legend()
        return figure

    def draw_spiral_contours(self, axes):
        mapper = pyopal.objects.ffa_field_mapper.FFAFieldMapper()
        rmin, rmax = axes.get_ylim()
        mapper.r_points = numpy.linspace(rmin, rmax, 20)
        self.r_points = mapper.r_points
        for contour in self.spiral_contours:
            if "phi1" in contour:
                self.draw_cylindrical_spiral_polygon(axes, contour)
            else:
                mapper.draw_cylindrical_spiral_contour(axes, contour)

    def draw_cylindrical_spiral_polygon(self, axes, contour):
        """
        Draw a radially spiralling contour on axes
        - axes: matplotlib Axes object to draw on.
        - contour: dictionary (see default_contour for definitions)
        """
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        tan_d = math.tan(math.radians(contour["spiral_angle"]))
        r_points = [r for r in self.r_points]
        phi_points = [math.radians(contour["phi0"]) + \
                         tan_d*math.log(r/contour["r0"]) for r in r_points]
        phi_points += [math.radians(contour["phi1"]) + \
                         tan_d*math.log(r/contour["r0"]) for r in reversed(r_points)]
        phi_points = [math.degrees(phi) for phi in phi_points]
        r_points += [r for r in reversed(r_points)]

        axes.fill(phi_points, r_points,
                  linestyle=contour["linestyle"],
                  color=contour["colour"],
                  alpha=contour["alpha"])
        axes.text(phi_points[-1],
                  self.r_points[-1],
                  contour["label"],
                  va="top",
                  rotation="vertical",
                  color=contour["colour"])
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)

    def azimuthal_fig(self, job_name, range_phi, time_plot_phi = None, do_field_plot = False):
        figure = matplotlib.pyplot.figure(figsize=(20, 10))
        axes = figure.add_subplot(2, 2, 1)
        optimisation_tools.utils.utilities.setup_large_figure(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, 'phi', 'z', range_phi, None)        
        #axes.legend()

        axes = figure.add_subplot(2, 2, 2)
        optimisation_tools.utils.utilities.setup_large_figure(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, 'phi', 'r', range_phi, None)
        if do_field_plot:
            range_r = axes.get_ylim()
            self.plot_2d(axes.figure, axes, [0.0, 0.0], range_phi, range_r, [-self.b0, self.b0], "phi", "r", "bz")
            for orbit in self.orbit_list:
                orbit.plot_2d(axes, 'phi', 'r', range_phi, None)

        if self.spiral_contours != None:
            self.draw_spiral_contours(axes)
        axes = figure.add_subplot(2, 2, 3)
        optimisation_tools.utils.utilities.setup_large_figure(axes)
        self.plot_1d_orbit(axes, self.orbit_list[0], 'phi', self.azimuthal_field_plot, range_phi, None)

        if time_plot_phi != None:
            axes = figure.add_subplot(2, 2, 4)
            optimisation_tools.utils.utilities.setup_large_figure(axes)
            centre = [0.0, 0.0, 0.0, 0.0]
            range_x = [0.0, 25000]
            self.plot_1d(axes, centre, range_x, [-0.1, 0.1], 't', ['btot'], None, 'time')

        figure.suptitle(job_name)
        return figure


    def cartesian_fig(self, job_name, range_y, orientation):
        figure = matplotlib.pyplot.figure(figsize=(20, 10))
        axes = figure.add_subplot(2, 2, 1) 
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, 'y', 'z', range_phi, None, orientation)
        axes.legend()

        axes = figure.add_subplot(2, 2, 2) 
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, 'y', 'x', range_phi, None)
        axes.legend()

        axes = figure.add_subplot(2, 2, 3)
        self.plot_1d_orbit(axes, self.orbit_list[0], 'y', ['bx', 'by', 'bz'], range_y, None)

        figure.suptitle(job_name)
        return figure

    def rf_fig(self, job_name, centre, time_list, range_t):
        figure = matplotlib.pyplot.figure(figsize=(20, 10))
        figure.suptitle(job_name)
        axes = figure.add_subplot(1, 1, 1) 
        for t_offset in time_list:
            self.plot_1d(axes, centre, range_t, [-1.0, 1.0], "t", ["ephi", "er", "btot"], t_offset, "")

    def rf_fig_2(self, job_name, centre, range_x, range_y):
        figure = matplotlib.pyplot.figure(figsize=(20, 10))
        figure.suptitle(job_name)
        axes = figure.add_subplot(2, 3, 1) 
        self.t0 = centre[3] 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.e0, self.e0], "x", "y", "ex")
        text = "t="+str(round(self.t0))+" ns"
        axes.text(0.8, 0.9, text, transform=axes.transAxes)
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        axes = figure.add_subplot(2, 3, 2) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.e0, self.e0], "x", "y", "ey")
        text = "t="+str(round(self.t0))+" ns"
        axes.text(0.8, 0.9, text, transform=axes.transAxes)
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        axes = figure.add_subplot(2, 3, 3) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.e0, self.e0], "x", "y", "ez")
        text = "t="+str(round(self.t0))+" ns"
        axes.text(0.8, 0.9, text, transform=axes.transAxes)
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)


        axes = figure.add_subplot(2, 3, 4) 
        self.plot_1d_orbit(axes, self.orbit_list[0], 'phi', ['ex', 'ey', 'ez'], [178.0, 182.0], [-self.e0, self.e0])
        axes = figure.add_subplot(2, 3, 5) 
        text = "t="+str(round(self.t0))+" ns"
        axes.text(0.8, 0.9, text, transform=axes.transAxes)
        self.plot_1d(axes, centre, [0., 1000.], [-10, 10], 't', ['ex', 'ey', 'ez'], 0.0, "rf")
        axes = figure.add_subplot(2, 3, 6) 
        self.plot_2d(figure, axes, centre, range_x, range_y, [-self.e0, self.e0], "x", "y", "bz")
        text = "t="+str(round(self.t0))+" ns"
        axes.text(0.8, 0.9, text, transform=axes.transAxes)
        if self.polygon_plotter != None:
            self.polygon_plotter.plot(axes)
        if self.log_plotter != None:
            self.log_plotter.plot(axes)
        for orbit in self.orbit_list:
            orbit.plot_2d(axes, "x", "y", range_x, range_y)

        return figure


    def bump_fig(self, job_name, radius, phi_list, range_t, range_phi):
        figure = matplotlib.pyplot.figure(figsize=(20, 10))
        figure.suptitle(job_name)
        axes = figure.add_subplot(1, 2, 1) 
        for phi in phi_list:
            name = "phi: "+format(int(math.degrees(phi)))+"$^\\circ$"
            centre = [radius*math.cos(phi), radius*math.sin(phi), 0, 0]
            self.plot_1d(axes, centre, range_t, [-1.0, 1.0], "t", ["br", "bz"], 0, name)
        axes = figure.add_subplot(1, 2, 2) 
        for orbit in self.orbit_list:
            orbit.plot_phi_z(axes)
        axes.set_xlim(range_phi)
        axes.set_ylim([-0.23, -0.07])
        #self.traj_fig(axes)
        for orbit in self.orbit_list:
            orbit.plot_phi_z(axes)
        return figure

    def traj_fig(self, axes):
        min_z, max_z = -0.1, 0.1
        var_x = "phi"
        field_variables = ["br", "bz", "bphi"]
        phi_list = self.orbit_list[0].orbit.orbit["phi"]
        r_list = self.orbit_list[0].orbit.orbit["r"]
        min_elev, max_elev = axes.get_ylim()
        min_phi, max_phi = axes.get_xlim()
        step_phi = (max_phi-min_phi)/self.n_2d_points
        step_elev = (max_elev-min_elev)/self.n_2d_points
        track_index = 0
        x_list, y_list, z_list = [], [], []
        for i in range(self.n_2d_points-1):
            phi = step_phi*(i+0.5)+min_phi
            while phi < phi_list[track_index] and track_index < len(phi_list)-1:
                track_index += 1
            radius = r_list[track_index]
            x = math.cos(phi)*radius
            y = math.sin(phi)*radius
            for j in range(self.n_2d_points-1):
                elev = step_elev*(j+0.5)+min_elev
                z = self.fields.get("br", x, y, elev, 0.0)
                if z < min_z:
                    z = min_z
                if z > max_z:
                    z = max_z
                x_list.append(phi)
                y_list.append(elev)
                z_list.append(z)
        cmax = max(abs(min_z), abs(max_z))
        vtot = max([abs(min(z_list)), max(z_list)])
        vtot = cmax #min(vtot, cmax)
        hist = axes.hist2d(x_list, y_list, self.n_2d_points, [[min_phi, max_phi], [min_elev, max_elev],], False, z_list, 
                    cmin=min_z, cmax=max_z, cmap=self.cmap, vmin=-vtot, vmax=vtot)
        axes.get_figure().colorbar(hist[3], ax=axes)

    def plot_1d(self, axes, centre, range_x, range_y, var_x, var_y_list, offset_x, name):
        i_x = self.pos_vars.index(var_x)
        min_x, max_x = range_x[0], range_x[1]
        min_y, max_y = range_y[0], range_y[1]
        pos = copy.deepcopy(centre)
        step_x = (max_x-min_x)/self.n_1d_points
        x_list = [step_x*(i+0.5)+min_x for i in range(self.n_1d_points-1)]
        for var_y in var_y_list:
            y_list = []
            for i, x in enumerate(x_list):
                pos[i_x] = x+offset_x
                y_value = self.fields.get(var_y, pos[0], pos[1], pos[2], pos[3])
                y_value = min(y_value, max_y)
                y_value = max(y_value, min_y)
                y_list.append(y_value)
            label = Labels.labels[var_y]
            if name:
                label += " "+name
            if offset_x:
                label += " offset "+format(offset_x, "6.4g")
            axes.plot(x_list, y_list, label=label)
            axes.set_xlabel(Labels.labels[var_x])
        axes.legend()

    def plot_2d(self, figure, axes, centre, range_x, range_y, range_z, var_x, var_y, var_z):
        min_x, max_x = range_x[0], range_x[1]
        min_y, max_y = range_y[0], range_y[1]
        min_z, max_z = range_z[0], range_z[1]
        pos = copy.deepcopy(centre)
        step_x = (max_x-min_x)/self.n_2d_points
        step_y = (max_y-min_y)/self.n_2d_points
        x_list = []
        y_list = []
        z_list = []
        for i in range(self.n_2d_points-1):
            x = step_x*(i+0.5)+min_x
            for j in range(self.n_2d_points-1):
                y = step_y*(j+0.5)+min_y
                x_list.append(x)
                y_list.append(y)
        point_list = self.get_points(var_x, var_y, x_list, y_list)
        for point in point_list:
            z = self.fields.get(var_z, point[0], point[1], point[2], point[3])
            if z > max_z:
                z = max_z
            if z < min_z:
                z = min_z
            z_list.append(z)
        cmax = max(abs(min_z), abs(max_z))
        vtot = max([abs(min(z_list)), max(z_list)])
        vtot = cmax #min(vtot, cmax)
        hist = axes.hist2d(x_list, y_list, self.n_2d_points, [[min_x, max_x], [min_y, max_y]], False, z_list, 
                    cmin=min_z, cmax=max_z, cmap=self.cmap, vmin=-vtot, vmax=vtot)
        axes.set_xlabel(Labels.labels[var_x])
        axes.set_ylabel(Labels.labels[var_y])
        matplotlib.pyplot.text(1.25, 0.9, Labels.labels[var_z], rotation=90, transform=axes.transAxes)
        if self.do_pipe:
            self.cartesian_lines(axes)
        figure.colorbar(hist[3], ax=axes)

    def get_points(self, var_x, var_y, x_list, y_list):
        point_list = [[0., 0., 0., 0.] for x in x_list]
        if var_x in self.pos_vars and var_y in self.pos_vars:
            i_x = self.pos_vars.index(var_x)
            i_y = self.pos_vars.index(var_y)
            for i, point in enumerate(point_list):
                point[i_x] = x_list[i]
                point[i_y] = y_list[i]
            return point_list
        elif var_x == "phi" and var_y == "r":
            for i, point in enumerate(point_list):
                point[0] = y_list[i]*math.cos(math.radians(x_list[i]))
                point[1] = y_list[i]*math.sin(math.radians(x_list[i]))
            return point_list
        else:
            raise RuntimeError("Failed to parse variables", var_x, var_y)

    def plot_1d_orbit(self, axes, orbit, var_x, var_y_list, range_x, range_y):
        i_list = [i for i, x in enumerate(orbit.orbit.orbit[var_x]) \
                                          if x < range_x[1] and x > range_x[0]]
        x_list = [orbit.orbit.orbit[var_x][i] for i in i_list]
        xcoord_list = orbit.orbit.orbit['x']
        ycoord_list = orbit.orbit.orbit['y']
        zcoord_list = orbit.orbit.orbit['z']
        y_dict_of_lists = dict([(var_y, []) for var_y in var_y_list])
        for i in i_list:
            x, y, z = xcoord_list[i], ycoord_list[i], zcoord_list[i]
            for var_y in var_y_list:
                y_dict_of_lists[var_y].append(self.fields.get(var_y, x, y, z, self.t0))
        for var_y, y_list in y_dict_of_lists.items():
            xy_list_of_lists = orbit.do_line_wraps(x_list, y_list)
            my_label = Labels.labels[var_y]
            for x_sublist, y_sublist in xy_list_of_lists:
                axes.plot(x_sublist, y_sublist, color=Labels.get_colors()[var_y], label=my_label)
                my_label = None
        axes.legend()
        axes.set_xlabel(Labels.labels[var_x])
        axes.set_title(orbit.name)
        if range_x != None:
            axes.set_xlim(range_x)
        if range_y != None:
            axes.set_ylim(range_y)

    def cartesian_lines(self, axes):
        for i_phi in range(self.n_phi):
            phi = math.pi*2.0/self.n_phi*i_phi
            x = self.r_min*math.cos(phi), self.r_max*math.cos(phi)
            y = self.r_min*math.sin(phi), self.r_max*math.sin(phi)
            axes.plot(x, y, color="gray")
        phi_list = [math.radians(i_phi) for i_phi in range(361)]
        for r0 in self.r_min, self.r_max:
            x = [r0*math.cos(phi) for phi in phi_list]
            y = [r0*math.sin(phi) for phi in phi_list]
            axes.plot(x, y, color="gray")

    pos_vars = ["x", "y", "z", "t"]

class PlotH5(object):
    def __init__(self, h5_file, dataset_name=""):
        self.h5 = h5_file
        self.s = optimisation_tools.utils.utilities.matplot_marker_size(self.h5.data)
        self.name = dataset_name

    def plot_phase_space(self, axes, x_axis, y_axis, z_axis, station):
        name = "phase_space_"+str(station).rjust(3, "0")+"_"+str(x_axis)+"_"+str(y_axis)
        plot_data = [item for item in self.h5.data if item["station"] == station]
        if len(plot_data) == 0:
            return
        x_list = [item[x_axis] for item in plot_data]
        y_list = [item[y_axis] for item in plot_data]
        scat = axes.scatter(x_list, y_list, c="blue", s=self.s, label=self.name)
        axes.set_xlabel(Labels.labels[x_axis])
        axes.set_ylabel(Labels.labels[y_axis])
        if z_axis != None:
            axes.get_figure().colorbar(scat)

    def phase_space_display(self, station):
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        z_axis = None
        axes = figure.add_subplot(2, 3, 1)
        self.plot_phase_space(axes, "x", "z", z_axis, station)
        axes = figure.add_subplot(2, 6, 7)
        self.plot_phase_space(axes, "x", "x'", z_axis, station)
        axes = figure.add_subplot(2, 6, 8)
        self.plot_phase_space(axes, "z", "z'", z_axis, station)
        axes = figure.add_subplot(2, 3, 2)
        self.plot_phase_space(axes, "u", "v", z_axis, station)
        axes = figure.add_subplot(2, 6, 9)
        self.plot_phase_space(axes, "u", "u'", z_axis, station)
        axes = figure.add_subplot(2, 6, 10)
        self.plot_phase_space(axes, "v", "v'", z_axis, station)
        axes = figure.add_subplot(2, 3, 3)
        self.plot_phase_space(axes, "au", "av", z_axis, station)
        axes = figure.add_subplot(2, 6, 11)
        self.plot_phase_space(axes, "phiu", "au", z_axis, station)
        axes = figure.add_subplot(2, 6, 12)
        self.plot_phase_space(axes, "phiv", "av", z_axis, station)

class LoadH5(object):
    def __init__(self, file_name_glob, time_window = None, verbose = 1, is_clockwise = False, will_load = True):
        self.mass = xboa.common.pdg_pid_to_mass[2212]
        self.time_window = time_window
        self.height_window= [-100.0, 100.]
        self.id_cut = [0]
        self.id_include = None
        self.file_name_glob = file_name_glob
        self.data = []
        self.f_glob = sorted(glob.glob(self.file_name_glob))
        self.station_count = len(self.f_glob)
        self.station_ids = dict((fname, index) for index, fname in enumerate(self.f_glob))
        self.verbose = verbose
        self.direction = 1
        self.station_dt = 1.0e-9 # if two hits in the same station and same id have dt < station_dt, discard the second hit
        if is_clockwise:
            self.direction = -1
        if self.verbose > 0:
            print ("Globbing ", self.file_name_glob, "yields", self.f_glob)
        if will_load:
            self.load_probes()

    def load_probes(self):
        for station, file_name in enumerate(self.f_glob):
            try:
                self.load_h5_probe(file_name)
            except Exception:
                sys.excepthook(*sys.exc_info())
                print("Error while opening", file_name)

    def load_h5_probe(self, file_name, station_list = None):
        # BUG - why is this so slow!
        data = []
        if self.verbose > 10:
            print("Loading h5 file", file_name, end=" ")
        h5_file = h5py.File(file_name, 'r')
        station_counter = {}
        last_time = {} # t of last hit from particle with id#
        for key in h5_file.keys():
            if key[:5] != "Step#" and self.verbose > 20:
                print("skipping key", key, end=" ")
                continue
            if self.verbose > 20:
                print("    ... ... ...", key)
            n_steps = len(h5_file[key]["x"])
            h5_step = h5_file[key]
            fdict = {}
            # convert from on disk to in-memory
            var_tuple = ("id", "x", "y", "z", "time", "px", "py", "pz")
            for var in var_tuple:
                h5_var = numpy.array(h5_step[var])
                fdict[var] = [x for x in h5_var]
            h5_step = fdict
            for i in range(n_steps):
                time =  h5_step["time"][i]
                hit_id = h5_step["id"][i]
                if hit_id not in station_counter:
                    station_counter[hit_id] = self.station_ids[file_name]
                    last_time[hit_id] = time
                else:
                    if abs(time - last_time[hit_id]) < self.station_dt:
                        continue
                    last_time[hit_id] = time
                    station_counter[hit_id] += self.station_count
                px = h5_step["px"][i]*self.mass
                py = h5_step["py"][i]*self.mass
                pz = h5_step["pz"][i]*self.mass
                phi = math.atan2(h5_step["y"][i], h5_step["x"][i])
                pphi = (-px*math.sin(phi) + py*math.cos(phi))*self.direction
                energy = (px**2+py**2+pz**2+self.mass**2)**0.5-self.mass
                item = {
                    "id":hit_id,
                    "station":station_counter[hit_id],
                    "x":h5_step["x"][i],
                    "y":h5_step["y"][i],
                    "z":h5_step["z"][i],
                    "t":time,
                    "phi":math.degrees(phi),
                    "r":(h5_step["x"][i]**2+h5_step["y"][i]**2)**0.5,
                    "x'":px/pphi,
                    "y'":py/pphi,
                    "z'":pz/pphi,
                    "r'":(px*math.cos(phi) + py*math.sin(phi))/pphi,
                    "phi'":(px*math.sin(phi) - py*math.cos(phi))/pphi,
                    "px":px,
                    "py":py,
                    "pz":pz,
                    "pphi":pphi,
                    "pr":(px*math.cos(phi) + py*math.sin(phi)),
                    "energy":energy
                }
                self.phi_domain(item)
                data.append(item)
                if self.verbose > 20:
                    print("H5", [(var, format(item[var], "4.5g")) for var in ('r', 'phi', 'pr', 'pphi')])
        if self.verbose > 10:
            print("... found", len(data), "points and after cuts", end=" ")
        data = self.cut(data)
        if self.verbose > 10:
            print("...", len(data), "points")
        self.data += data

    def set_closed_orbit(self, closed_orbit):
        for item in self.data:
            #for item in datum:
                ref = closed_orbit.get_ref_track(item["station"])
                if not ref:
                    item.update({"u":0.0, "up":0.0, "v":0.0, "vp":0.0, 
                                 "phiu":0.0, "au":0.0, "phiv":0.0, "av":0.0, "a4d":0.0})
                    continue
                coupled = [item["r"]*1e3 - ref["x"], ref["px"]/ref["pz"]-item["r'"], item["z"]*1e3 - ref["y"], ref["y'"] - item["z'"]]
                decoupled = closed_orbit.tm.decoupled(coupled).tolist()
                aa = closed_orbit.tm.coupled_to_action_angle(coupled)
                aa[1] *= ref["p"]/self.mass
                aa[3] *= ref["p"]/self.mass
                item.update({"u":decoupled[0], "u'":decoupled[1],
                             "v":decoupled[2], "v'":decoupled[3], 
                             "phiu":aa[0], "au":aa[1],
                             "phiv":aa[2], "av":aa[3], "a4d":aa[1]+aa[3]})

    @classmethod
    def phi_domain(cls, item):
        dphi = cls.azimuthal_domain[1]-cls.azimuthal_domain[0]
        while item["phi"] < cls.azimuthal_domain[0]:
                    item["phi"] += dphi
        while item["phi"] > cls.azimuthal_domain[1]:
                    item["phi"] -= dphi

    def cut(self, data):
        if self.id_cut:
            data = [item for item in data if item["id"] not in self.id_cut]
        if self.id_include:
            data = [item for item in data if item["id"] in self.id_include]
        if self.time_window and len(data):
            start_time = min([item["t"] for item in data])
            data = [item for item in data if item["t"] < start_time+self.time_window]
        if self.height_window:
            data = [item for item in data if item["z"] < self.height_window[1] and item["z"] > self.height_window[0]]
        return data

    azimuthal_domain = [-180.0, 180.0]

class LoadClosedOrbit(object):
    def __init__(self, file_name, tm_entry=None, tm_is_decoupled=False):
        self.ref_track = []
        self.tm = None
        self.file_name = file_name
        self.tm_is_decoupled = True
        self.tm_entry = tm_entry # take the i^th entry in tm_list (for multiple cells)
        self.pid = 2212
        self.load_closed_orbits()

    def load_closed_orbits(self):
        fin = open(self.file_name)
        co = json.loads(fin.readline())[0]
        self.ref_track = []
        for hit in co["ref_track"]:
            mass = xboa.common.pdg_pid_to_mass[self.pid]
            hit.update({"pid":self.pid, "mass":mass, "charge":1.})
            hit = xboa.hit.Hit.new_from_dict(hit)
            self.ref_track.append(hit)
        if self.tm_entry is None:
            tm = co["tm"]
        else:
            tm = co["tm_list"][self.tm_entry]
        tm = [row[1:5] for row in tm]
        if self.tm_is_decoupled:
            self.tm = TwoDTransferMatrix(tm, True)
        else:
            self.tm = DecoupledTransferMatrix(tm, True)

    def get_ref_track(self, station):
        try:
            return self.ref_track[station]
        except IndexError:
            return None

class PlotOrbit(object):
    def __init__(self, orbit, name="", probe_data=None):
        self.orbit = orbit
        self.probe = probe_data
        self.name = name
        self.r0 = None

    def do_plots(self):
        self.plot_phi_z()

    def plot_phi_z(self, axes):
        self.plot_2d(axes, "phi", "z", None, None)

    def plot_probe(self, axes, var_x, var_y, range_x, range_y, color):
        x_list = [item[var_x] for item in self.probe.data]
        y_list = [item[var_y] for item in self.probe.data]
        #print("Plotting probes", x_list, y_list)
        axes.scatter(x_list, y_list, label=self.name, s=20, marker="o", facecolors="none", edgecolors=color)

    def do_line_wraps(self, x_list, y_list):
        if not len(x_list):
            return [[x_list, y_list]]
        delta_x = max(x_list)-min(x_list)
        break_index_list = []
        for i, x in enumerate(x_list[1:]):
            if abs(x-x_list[i]) > delta_x/2.0:
                break_index_list.append(i)
        break_index_list.append(-1)
        print("Detected line wraps at:", break_index_list)
        prev_index = -1
        xy_list_of_lists = []
        for i, next_index in enumerate(break_index_list):
            x_sublist = x_list[prev_index+1:next_index]
            y_sublist = y_list[prev_index+1:next_index]
            xy_list_of_lists.append( (x_sublist, y_sublist) )
            prev_index = next_index
        return xy_list_of_lists

    def plot_2d(self, axes, var_x, var_y, range_x, range_y):
        i_list = range(len(self.orbit.orbit[var_x]))
        if range_x != None:
            i_list = [i for i in i_list \
                if self.orbit.orbit[var_x][i] < range_x[1] and self.orbit.orbit[var_x][i] > range_x[0]] 
        x_list = [self.orbit.orbit[var_x][i] for i in i_list]
        y_list = [self.orbit.orbit[var_y][i] for i in i_list]
        if len(x_list) == 0:
            print("Failed to find any points for orbit plot from file name", self.orbit.file_name)
            return
        axes.set_xlabel(Labels.labels[var_x])
        axes.set_ylabel(Labels.labels[var_y])
        if range_x != None:
            axes.set_xlim(range_x)
        if range_y != None:
            axes.set_ylim(range_y)
        xy_list_of_lists = self.do_line_wraps(x_list, y_list)
        for x_sublist, y_sublist in xy_list_of_lists:
            line2d = axes.plot(x_sublist, y_sublist, label=self.name)
        if self.probe != None:
            self.plot_probe(axes, var_x, var_y, range_x, range_y, color=line2d[0].get_color())
        if var_x == "phi" and self.r0:
            limits = axes.get_xlim()
            axes = axes.twiny()
            axes.set_xlim([limits[0]*self.r0*2.0*math.pi/360.0, limits[1]*self.r0*2.0*math.pi/360.0])
            axes.set_xlabel(f"$r_0\\phi$ [m] for $r_0$ = {self.r0} m")

class PlotPolygon(object):
    def __init__(self, n_cells, cell_length):       
        self.n_cells = n_cells
        self.cell_length = cell_length
        self.start_theta = math.pi*2.0/n_cells/2.0
        self.start_x = self.cell_length/2.0/math.sin(self.start_theta)
        self.start_y = 0.0


    def plot(self, axes):
        x_list, y_list = [self.start_x], [self.start_y]
        theta = self.start_theta
        for i in range(self.n_cells):
            x_list.append(x_list[-1]-self.cell_length*math.sin(theta))
            y_list.append(y_list[-1]+self.cell_length*math.cos(theta))
            theta = theta + 2.0*math.pi/self.n_cells
        axes.plot(x_list, y_list, c="lightgrey")

class PlotUtils():
    def __init__(self):
        pass

    @classmethod
    def hist_range(cls, data_list):
        if len(data_list) < 2:
            raise ValueError("Not enough points in data_list"+str(data_list))
        data_list = sorted(list(set(data_list)))
        x_min = data_list[0]-(data_list[1]-data_list[0])/2
        x_max = data_list[-1]+(data_list[-1]-data_list[-2])/2
        n_points = len(data_list)
        return n_points, x_min, x_max



