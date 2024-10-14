import argparse
import glob
import json
import bisect
import copy
import os
import time
import math
import subprocess
import sys

import scipy
import matplotlib
import numpy
import xboa.bunch
import xboa.common

from optimisation_tools.utils import decoupled_transfer_matrix 
from optimisation_tools.utils import twod_transfer_matrix
decoupled_transfer_matrix.DecoupledTransferMatrix.det_tolerance=1
from optimisation_tools.utils import utilities
from optimisation_tools.toy_model.foil_model.material import Material
from optimisation_tools.toy_model.foil_model.particle import Particle

import optimisation_tools.analysis.rf_bucket

numpy.set_printoptions(edgeitems=30, linewidth=200)
P_MASS = 938.27208816


class ToyModel(object):
    def __init__(self):
        n_injection_turns = 1
        n_trajectory_turns = 50
        injection_amplitude = 1.3
        n_turns = n_injection_turns+n_trajectory_turns
        #Run 101 with args [0.2116535191136668, -1.1802958558145478, 1.5605421957991457]Y MAX 10.0
        # injection setup
        self.momentum = 75.
        self.rf_reference_momentum = 75*(1.002) # used to calculate RF frequency and plot RF buckets/etc
        self.max_turn = n_turns
        self.number_pulses = n_injection_turns
        self.number_per_pulse = 1000
        # default bumps = position of the proton orbit
        angle_u = -0.11628784278921322
        angle_v = 1.4012763546351992
        if n_injection_turns == 1:
            self.calculated_bumps = [["action_angle", angle_u, 0.0, angle_v, 0.0]] #"action_angle" or "coupled"
        else:
            self.calculated_bumps = [["action_angle",
                                   angle_u, i*injection_amplitude//(n_injection_turns-1), #(n_injection_turns-1.-i)*injection_amplitude/(n_injection_turns-1.),
                                   angle_v, i*injection_amplitude/(n_injection_turns-1)] for i in range(n_injection_turns)]
            inj_end = self.calculated_bumps[-1]
            traj_end = [angle_u, 5., angle_v, 5.]
            self.calculated_bumps += [[(traj_end[j]-inj_end[j])*i/(n_trajectory_turns)+inj_end[j] for j in range(4)] for i in range(1, n_trajectory_turns+1)]
        self.calculated_bumps = [["coupled", 0.0, 0.0, 0.0*i/(n_turns-1), 0.0] for i in range(n_turns)]
        self.calculated_injection = None # position of the injection orbit in aa coordinates
        self.closed_orbit_coords = [4357.790553052024, 0.0, 0.0, 0.0]
        self.foil_angle = math.degrees(math.pi)
        self.foil_de_for_rf_bucket = 0.0 # for use in RF bucket calculation only
        self.n_foil_sigma = 3 # number of sigma; set to < 0 to disable foil adjustment

        self.injection_ellipse_algorithm = "transfer_matrix" # transfer_matrix from_twiss or user_defined
        self.beta_x = 1.0 # beta [m]
        self.alpha_x = 0.0
        self.beta_y = 1.0 # beta [m]
        self.alpha_y = 0.0
        self.injection_ellipse = None

        self.m_index = 7.1 #-1.31 # m^-1 [vertical] or dimensionless k-value [horizontal]
        self.r0 = 4000.0
        self.pulse_emittance = 0.026
        self.dp_model = "gauss" # none gauss or fixed
        self.dp_over_p = 0.00132 # 0.0029 # sigma
        self.max_dp_over_p = self.dp_over_p*3 # only used at injection time
        self.foil_material = "carbon"
        self.foil_column_density = 20e-6 # g/cm^2
        self.pid = 2212
        self.amplitude_acceptance = 0.030 # mm
        self.dp_over_p_acceptance = 0.004 # +/- dp/p

        self.beam_pulses = [[0.5-150.0/979.2889189321706, 0.5+150.0/979.2889189321706]] # time pulses relative to ring tof
        self.harmonic_number = 2.0 # rf frequency = harmonic_number/ring_tof
        self.rf_voltage = 0.004 # rf peak voltage [MV/turn]
        self.rf_time_offset = 0.0 # rf time offset
        self.momentum_offset_injection = True # assume dispersion matching for injected beam

        # execution parameters
        self.lattice = "test"
        self.study = "testing"
        self.version = 1
        self.verbose = 8
        self.verbose_particles = [0]
        self.n_cells = 15 # number of cells for transverse tm
        self.tof_multiplier = 16 # ring tof = cell tof * tof_multiplier
        self.accumulate = False
        self.do_plots = True
        self.plot_frequency = 5 #self.max_turn # controls whether to plot frames
        self.do_stats = True
        self.stats_frequency = self.max_turn # controls whether to plot frames
        self.do_movie = False
        self.sleep_time = 0.01 #0.1
        self.f_size = 20
        self.l_size = 14
        self.do_scattering = True
        self.do_energy_loss = True
        self.seed = None
        self.is_horizontal = True # horizontal excursion or vertical
        self.beam_filename = "beam_at_end.txt"

        self.real_range = [25.0*self.amp_scale, 0.05*self.amp_scale, 25.0*self.amp_scale, 0.025*self.amp_scale]
        self.real_centre = [0.0*self.amp_scale, 0.0*self.amp_scale, 0.0*self.amp_scale, 0.0*self.amp_scale]

        self.dec_range = [25.0*self.amp_scale, 0.05*self.amp_scale, 25.0*self.amp_scale, 0.025*self.amp_scale]
        self.dec_centre = [0.0*self.amp_scale, 0.0*self.amp_scale, 0.0*self.amp_scale, 0.0*self.amp_scale]

        self.aa_range = [200, 0.02, 200, 0.02]
        self.aa_centre = [0., 0.02, 0., 0.02]

        # internal data
        self.bump_fields = []
        self.bump_orbits = []
        self.bump_tms = []
        self.mass = xboa.common.pdg_pid_to_mass[self.pid]
        self.beta_rel = self.momentum / (self.momentum**2+self.mass**2)**0.5 # beta = p/E
        self.foil_edge = -1
        self.rf_angular_frequency = 0.0
        self.ring_tof = 0.0 # ring time of flight at nominal momentum self.momentum
        self.circumference = 0.0 # assumed momentum independent (vffa)
        self.beam_data = numpy.empty_like([], shape=(0, 4))
        self.injection_orbits = []
        self.in_longitudinal_acceptance = []
        self.in_transverse_acceptance = []
        self.dp_over_p_data = []
        self.t_data = []
        self.beam_injection_turn = []
        self.turn = 0
        self.foil_hits = [] # number of foil hits per particle
        self.foil_hit_positions = [] # positions of foil hits
        self.first_turn_positions = [] # positions of foil hits
        self.rf_bucket_contours = None
        self.output_dir = ""
        self.is_two_d = True
        self.reset_output()
        self.setup_material()
        self.setup_subplots()

    def reset_output(self):
        self.output = {
            "mean_foil_hits":0,
            "max_foil_hits":0,
            "rms_dp_over_p":0,
            "dp_over_p_1e-3":0,
            "dp_over_p_1e-2":0,
            "amplitude_4d_1e-3":0,
            "amplitude_u_1e-3":0,
            "amplitude_v_1e-3":0,
            "amplitude_4d_1e-2":0,
            "amplitude_u_1e-2":0,
            "amplitude_v_1e-2":0,
            "rms_emittance_4d":0,
            "rms_emittance_u":0,
            "rms_emittance_v":0,
            "hits_per_turn":[],
            "beam_trajectory":[],
            "n_outside_acceptance":0,
        }


    def _norm(self):
        """conversion factor from normalised emittance coordinates
        Normalised emittance = beta*gamma*geometric_emittance
        To convert from normalised to geometric emittance MULTIPLY by norm
        """
        return self.momentum/P_MASS

    def geo_to_norm(self, geometric_emittance):
        return geometric_emittance*self.momentum/P_MASS

    def norm_to_geo(self, norm_emittance):
        return norm_emittance*P_MASS/self.momentum


    def do_config(self, config):
        for key in config:
            if key in self.__dict__:
                self.__dict__[key] = config[key]
            else:
                raise(KeyError("Did not recognise config key '"+str(key)+"'"))
        self.setup_material()
        for fig in [self.fig, self.fig2, self.fig3, self.fig4]:
            utilities.plot_id(fig, self.lattice, self.study, self.version)
        #self.output_dir += utilities.directory_name("", self.study, self.version)
        if self.seed != None:
            numpy.random.seed(self.seed)

    def foil_edge_from_beam(self, n_sigma):
        ellipse = self.get_injection_ellipse()
        ellipse = numpy.array([
                [ellipse[0][0], ellipse[2][0],],
                [ellipse[2][0], ellipse[2][2],],
            ])
        theta = math.radians(self.foil_angle)
        R = numpy.array([
                [math.cos(theta), math.sin(theta)],
                [-math.sin(theta), math.cos(theta)]
            ])
        ellipse = numpy.dot(R, ellipse)
        ellipse = numpy.dot(ellipse, R.transpose())
        self.foil_edge = max(ellipse[1][1]**0.5*n_sigma, 0.1)

    def beam_from_foil_edge(self):
        theta = math.radians(self.foil_angle)
        R = numpy.array([
                [math.cos(theta), math.sin(theta)],
                [-math.sin(theta), math.cos(theta)]
            ])

        centre = numpy.mean(self.beam_data, 0)[0:-1:2]
        centre = numpy.dot(R, centre)

        ellipse = numpy.cov(self.beam_data, rowvar=False)
        ellipse = numpy.array([[ellipse[0][0], ellipse[2][0]],
                               [ellipse[0][2], ellipse[2][2]],])
        ellipse = numpy.dot(R, ellipse)
        ellipse = numpy.dot(ellipse, R.transpose())

        distance = (centre[1]-self.foil_edge)
        if ellipse[1][1] > 0.:
            n_sigma = distance/ellipse[1][1]**0.5
        else:
            n_sigma = numpy.nan

        return distance, n_sigma

    def setup_material(self):
        self.foil = Material()
        self.foil.set_material(self.foil_material)
        self.foil_thickness = self.foil_column_density/self.foil.density

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default="")
        parser.add_argument("--bump_settings", type=str, default="")
        parser.add_argument("--closed_orbit", type=str, default="")
        parser.add_argument("--foil_station", type=int, default=0)
        args = parser.parse_args()
        self.settings = args.bump_settings
        self.closed_orbit = args.closed_orbit
        self.foil_station = args.foil_station
        self.output_dir = args.output_dir+"/"

    def initialise(self):
        self.load_closed_orbit()
        self.load_bump_settings()
        self.foil_edge_from_beam(self.n_foil_sigma)
        if self.verbose > 4:
            tm = self.bump_tms[0]
            print("Using transfer map with determinant", numpy.linalg.det(tm.m), " m:")
            print(numpy.real(tm.m))
            print("t:")
            print(numpy.real(tm.t))
            print(numpy.linalg.det(tm.t[0:2, 0:2]), numpy.linalg.det(tm.t[2:4, 2:4]), numpy.linalg.det(tm.t),)
            print("nu:")
            print(round(tm.get_phase_advance(0)/math.pi/2., 4), round(tm.get_phase_advance(1)/math.pi/2., 4))
            print("r:")
            print(numpy.real(tm.r))
            print(numpy.linalg.det(tm.r))
            print("v_t:")
            print(numpy.real(tm.v_t))
            print(numpy.linalg.det(tm.v_t[0:2, 0:2]), numpy.linalg.det(tm.v_t[2:4, 2:4]), numpy.linalg.det(tm.v_t))
            print("Closed orbits:")
            for i, turn in enumerate(range(self.max_turn)):
                orbit = self.bump_orbits[turn]
                o_str = ' '.join([format(x, "8.4g") for x in orbit])
                aa_str = ' '.join([format(x, "8.4g") for x in tm.coupled_to_action_angle(orbit-self.injection_orbits[0])])
                dec_str = ' '.join([format(x, "8.4g") for x in tm.decoupled(orbit-self.injection_orbits[0])])
                print("  Orbit", i, o_str, "  action angle", aa_str, "  decoupled", dec_str)
        self.reset_output()
        self.build_bucket(will_replace = True)
        self.beam_data = numpy.empty_like([], shape=(0, 4))
        self.turn = 0
        self.beam_injection_turn = []
        self.foil_hits = []
        self.foil_hit_positions = []
        self.first_turn_positions = []
        self.dp_over_p_data = []
        self.in_longitudinal_acceptance = []
        self.in_transverse_acceptance = []
        self.t_data = []

    def load_closed_orbit(self):
        if self.verbose > 3:
            print("Loading closed orbits", self.closed_orbit)
        closed_orbit = json.loads(open(self.closed_orbit).readline())
        tm = closed_orbit[0]["tm"]
        cell_time = closed_orbit[0]["ref_track"][1]["t"]
        one_cell = numpy.array([row[1:5] for row in tm])
        if self.is_two_d:
            if self.verbose > 4:
                print("Using 2D Transfer Matrix")
            dm = twod_transfer_matrix.TwoDTransferMatrix(one_cell)
        else:
            dm = decoupled_transfer_matrix.DecoupledTransferMatrix(one_cell)
        if self.verbose > 4:
            print("Loading one cell transfer matrix with decoupled tunes", 
                  round(dm.get_phase_advance(0)/math.pi/2., 4),
                  round(dm.get_phase_advance(1)/math.pi/2., 4),
                  "and converting it to one turn matrix using", self.n_cells, "cells")
        tm = copy.deepcopy(one_cell)
        for i in range(self.n_cells-1):
            tm = numpy.dot(tm, one_cell)
        if self.is_two_d:
            tm = twod_transfer_matrix.TwoDTransferMatrix(tm)
        else:
            tm = decoupled_transfer_matrix.DecoupledTransferMatrix(tm)
        if self.verbose > 4:
            print("Transfer matrix determinant", numpy.linalg.det(tm.m))
        # ring tof comes from the closed orbit calculation
        self.ring_tof = cell_time*self.tof_multiplier
        # circumference comes from the closed orbit and assumed CO momentum
        self.circumference = self.ring_tof * self.beta_rel * xboa.common.constants["c_light"]
        # RF cavities can be set for a different central p to the injected beam.
        rf_beta = self.rf_reference_momentum/(self.rf_reference_momentum**2+self.mass**2)**0.5
        rf_tof = self.circumference/rf_beta/xboa.common.constants["c_light"]
        self.rf_angular_frequency = 2.0*math.pi*self.harmonic_number/rf_tof
        if self.verbose > 0:
            print("Found cell tof", cell_time, "ns and ring tof in", self.n_cells, "cells", self.ring_tof, "ns")

        self.bump_tms  = [tm]*(self.max_turn+1)

    def trajectory(self, trajectory_list):
        trajectory  = [None]*(self.max_turn+1)

        for i in range(self.max_turn+1):
            if i < len(trajectory_list):
                definition = trajectory_list[i]
            else:
                definition = trajectory_list[-1]
            if definition[0] == "action_angle":
                tm = self.bump_tms[i]
                action_angle_bump = definition[1:]
                action_angle_bump[1] = self.norm_to_geo(action_angle_bump[1])
                action_angle_bump[3] = self.norm_to_geo(action_angle_bump[3])
                coupled_bump = tm.action_angle_to_coupled(action_angle_bump) # expects geometric actions
                if self.verbose > 4:
                    print(f"bump: {definition} yields {coupled_bump}")
            elif definition[0] == "coupled":
                tm = self.bump_tms[i]
                coupled_bump = numpy.array(definition[1:])
            elif definition[0] == "delta":
                if i == 0:
                    raise ValueError("First bump setting can't be delta")
                tm = self.bump_tms[i]
                coupled_bump = numpy.array(definition[1:])+trajectory[i-1]
            else:
                raise ValueError("Did not recognose default bumps coordinate system"+\
                                 str(definition[0]))
            trajectory[i] = coupled_bump
        return trajectory

    def calculate_bump_trajectory(self):
        if self.verbose > 3:
            print("Calculating bump settings by hand")
        self.bump_orbits = self.trajectory(self.calculated_bumps)
        return

    def calculate_injection_trajectory(self):
        if self.calculated_injection == None:
            if self.verbose > 3:
                print("Assuming injection does not move")
            self.injection_orbits = [[0.0, 0.0, 0.0, 0.0] for i in range(self.max_turn+1)]
            return
        if self.verbose > 3:
            print("Calculating injection trajectory by hand")
        self.injection_orbits = self.trajectory(self.calculated_injection)
        return


    def injection_orbit_settings(self):
        self.injection_orbits  = []
        for i in range(self.max_turn+1):
            self.injection_orbits.append([0, 0, 0, 0])

    def load_bump_settings(self):
        self.calculate_injection_trajectory()
        if self.settings == "":
            self.calculate_bump_trajectory()
            return
        file_list = sorted(glob.glob(self.settings))
        if self.verbose > 3:
            print("Loading", len(file_list), "bump settings",
                  file_list[0], "...", file_list[-1], end=" ... ")
        for my_file in file_list:
            line_in = open(my_file).readlines()[-1]
            line_json = json.loads(line_in)
            if line_json["optimisation_stage"]["optimisation_stage"] != 1:
                continue
            for hit in line_json["tracking"]:
                if hit[0] == self.foil_station:
                    self.bump_fields.append(line_json["bump_fields"])
                    self.bump_orbits.append(numpy.array(hit[1:]))
                    break
        self.injection_orbits = [[0.0, 0.0, 0.0, 0.0] for b in self.bump_orbits]
        if self.verbose > 3:
            print("Done")

    def run_toy_model(self):
        print(f"Running toy model for {self.lattice} {self.study} {self.version}")
        self.setup_output_dir()
        try:
            if self.verbose > 3:
                print("Running")
            while True:
                self.do_one_turn()
        except StopIteration:
            pass

    def print_particles(self):
        targets = self.verbose_particles
        if targets == None:
            print("printing all particles")
            targets = range(len(self.beam_data))
        if len(targets) > 0:
            print("Printing particle data for turn", self.turn)
        for i in targets:
            print(i, "u:", self.beam_data[i], "t, p:", self.t_data[i], self.dp_over_p_data[i])

    def print_beam(self, filename):
        with open(filename, "w") as fout:
            for i in range(len(self.beam_data)):
                fout.write(f"{i} ")
                for item in self.beam_data[i]:
                    fout.write(f"{item} ")
                fout.write(f"{self.t_data[i]} {self.dp_over_p_data[i]}\n")


    def do_one_turn(self):
        self.inject_one() # inject on the injection orbit in global coordinates
        if self.verbose > 7:
            print("Turn", self.turn)
            print("  Bump orbit", self.get_bump_orbit())
            print("  injection ", self.injection_orbits)
        if self.verbose > 10:
            print(self.beam_data)
        self.print_particles()
        self.check_foil() # check for foil hits and apply foil physics
        self.do_longitudinal() # do longitudinal model for one turn (assume we do this after the first scatter/foil hit)
        self.turn_summary() # plots and stats
        self.print_particles()
        self.beam_data = self.transform_to_relative_coordinates(None, None) # transform to coordinates relative to momentum-dependent closed orbit
        self.print_particles()
        self.turn_one()  # do one turn transfer matrix phase advance
        self.update_turn() # add one to turn (index); note this moves the position of the bump orbit
        self.print_particles()
        self.beam_data = self.transform_to_absolute_coordinates(None, None) # transform to global coordinate system in new bumped coordinates
        self.print_particles()

    def print_aa(self):
        aa_hits = self.convert_to_aa(self.beam_data)
        print("Printing aa for turn", self.turn)
        for i, hit in enumerate(aa_hits):
            print("Print aa", i, hit)

    def build_ellipse(self):
        emit_x = self.norm_to_geo(self.pulse_emittance)
        emit_y = self.norm_to_geo(self.pulse_emittance)
        beta_x = self.beta_x*1000.0 # convert to mm
        beta_y = self.beta_y*1000.0 # convert to mm
        gamma_x = (1+self.alpha_x**2)/beta_x
        gamma_y = (1+self.alpha_y**2)/beta_y
        ellipse = [
            [       emit_x*beta_x, -emit_x*self.alpha_x,  0.0,             0.0],
            [-emit_x*self.alpha_x,  emit_x*gamma_x,       0.0,             0.0],
            [ 0.0,    0.0,        emit_y*beta_y, -emit_y*self.alpha_y],
            [ 0.0,    0.0, -emit_y*self.alpha_y,  emit_y*gamma_y],
        ]
        ellipse = numpy.array(ellipse)
        try:
            numpy.linalg.cholesky(ellipse)
        except:
            print("Oops")
            print(ellipse)
            sys.excepthook(*sys.exc_info())
        return ellipse


    def setup_output_dir(self):
        if self.output_dir == "":
            self.output_dir = os.path.split(self.closed_orbit)[0]+"/toy_model/"
        utilities.clear_dir(self.output_dir)

    def get_injection_ellipse(self):
        if self.injection_ellipse_algorithm == "transfer_matrix":
            return self.get_transfer_matrix().get_v_m([self.norm_to_geo(self.pulse_emittance)]*2)
        elif self.injection_ellipse_algorithm == "from_twiss":
            return self.build_ellipse()
        elif self.injection_ellipse_algorithm == "user_defined":
            return self.injection_ellipse
        else:
            raise ValueError("Did not recognise injection_ellipse_algorithm "+str(self.injection_ellipse_algorithm))

    def injecting(self):
        return len(self.beam_data) < self.number_pulses*self.number_per_pulse

    def time_distribution(self, number):
        time_bucket_list = [0.0]
        for i, item in enumerate(self.beam_pulses):
            time_bucket_list.append(time_bucket_list[-1]+item[1]-item[0])

        t_list_total = list(numpy.random.uniform(0.0, time_bucket_list[-1], number))
        t_list = []
        #print("Injecting into buckets", [(t[0]*self.ring_tof, t[1]*self.ring_tof) for t in self.beam_pulses])
        for t in t_list_total:
            for i, time_bucket_upper in enumerate(time_bucket_list[1:]):
                if t <= time_bucket_upper:
                    t_bucketed = t-time_bucket_list[i]+self.beam_pulses[i][0]
                    t_list.append(t_bucketed*self.ring_tof)

                    break
        self.t_data = self.t_data+t_list

    def delta_co(self, dp_over_p):
        mean = numpy.array([0, 0, 0, 0])
        if self.is_horizontal:
            mean[0] += (dp_over_p)**1/self.m_index*self.r0 # mm
        else:
            mean[2] += dp_over_p/self.m_index*1000. # mm
        return mean

    def inject_one(self):
        number = self.number_per_pulse
        events = numpy.zeros((number, 4))
        dp_over_p_list = [0 for i in range(number)]
        total_length = 0.0
        n_successes = 0
        ellipse = self.get_injection_ellipse()
        while n_successes < self.number_per_pulse:
            if self.dp_model == "gauss" and self.dp_over_p > 1e-9:
                dp_over_p = numpy.random.normal(0., self.dp_over_p)
            elif self.dp_model == "fixed" and self.number_per_pulse > 1:
                dp_over_p = self.dp_over_p*n_successes/(self.number_per_pulse-1)
            elif self.dp_model == "none":
                dp_over_p = 0.0
            else:
                raise RuntimeError("Failed to do dp_model "+str(self.dp_model))
            if dp_over_p > self.max_dp_over_p or dp_over_p < -self.max_dp_over_p:
                continue
            mean = copy.deepcopy(self.injection_orbits[self.turn])
            if self.momentum_offset_injection:
                mean += self.delta_co(dp_over_p)
            dp_over_p_list[n_successes] = dp_over_p
            psv = numpy.random.multivariate_normal(mean, ellipse, 1)
            if not self.inside_foil(psv[0]):
                continue
            events[n_successes] = psv
            n_successes += 1
        if not self.injecting():
            return
        self.time_distribution(number)
        self.beam_data = numpy.append(self.beam_data,
                                      events,
                                      axis=0)
        self.dp_over_p_data += dp_over_p_list
        self.beam_injection_turn += [self.turn for i in range(number)]
        self.first_turn_positions += [(hit[0], hit[2]) for hit in events]
        if self.verbose > 2:
            print("Injecting pulse with orbit", self.get_bump_orbit(self.turn, 0), self.injection_orbits[self.turn])


    def transform_to_relative_coordinates(self, beam_data, dp_over_p):
        # transform to coordinate system of bumped orbit
        if type(beam_data) == type(None):
            beam_data = self.beam_data
        for i, row in enumerate(beam_data):
            if dp_over_p == None:
                this_dp_over_p = self.dp_over_p_data[i]
            else:
                this_dp_over_p = float(dp_over_p)
            bump_orbit = self.get_bump_orbit(self.turn, this_dp_over_p)
            row -= bump_orbit
        return beam_data

    def transform_to_absolute_coordinates(self, beam_data, dp_over_p):
        # transform back to global coordinate system
        if type(beam_data) == type(None):
            beam_data = self.beam_data
        for i, row in enumerate(beam_data):
            if dp_over_p == None:
                this_dp_over_p = self.dp_over_p_data[i]
            else:
                this_dp_over_p = float(dp_over_p)
            bump_orbit = self.get_bump_orbit(self.turn, this_dp_over_p)
            row += bump_orbit
        return beam_data

    def turn_one(self):
        tm = numpy.real(self.get_transfer_matrix().m)
        self.beam_data = numpy.transpose(self.beam_data)
        self.beam_data = numpy.dot(tm, self.beam_data)
        self.beam_data = numpy.transpose(self.beam_data)

    def update_turn(self):
        if self.last_turn():
            raise StopIteration("turn "+str(self.turn)+" > max turn "+str(self.max_turn))
        self.turn += 1

    def last_turn(self):
        return self.turn >= self.max_turn

    def get_bump_orbit(self, turn = None, dp_over_p = 0):
        if turn == None:
            turn = self.turn
        orbit = copy.deepcopy(self.bump_orbits[self.turn])
        orbit += self.delta_co(dp_over_p)
        return orbit

    def get_bumper_index(self, turn = None):
        if turn == None:
            turn = self.turn
        return turn

        index = bisect.bisect_left(self.turn_bumper_index[0], turn)
        if index <= 0:
            return 0
        elif index >= len(self.turn_bumper_index[0]):
            return len(self.turn_bumper_index[0])-1
        turn_0 = self.turn_bumper_index[0][index-1]
        turn_1 = self.turn_bumper_index[0][index]
        index = (turn-turn_0)/float(turn_1-turn_0)+self.turn_bumper_index[1][index-1]
        return index

    def get_transfer_matrix(self, turn = None):
        return self.bump_tms[self.get_bumper_index(turn)]
        index = int(self.get_bumper_index(turn))
        bumper_setting = self.turn_bumper_index[1][index]

    def convert_to_decoupled(self, hits):
        r_mat = self.get_transfer_matrix().r_inv
        hits = numpy.dot(r_mat, numpy.transpose(hits))
        hits = numpy.transpose(numpy.real(hits))
        return hits

    def convert_to_aa(self, coupled_hits):
        norm = [180./math.pi, self.geo_to_norm(1.0), 180./math.pi, self.geo_to_norm(1.0)] # normalised action in microns
        tm = self.get_transfer_matrix()
        aa_hits = [tm.coupled_to_action_angle(hit) for hit in coupled_hits]
        aa_hits = [[u*norm[i] for i, u in enumerate(hit)] for hit in aa_hits]
        return aa_hits

    def do_longitudinal(self):
        for i, time in enumerate(self.t_data):
            p0 = self.momentum*(1+self.dp_over_p_data[i])
            E0 = (p0**2+self.mass**2)**0.5
            E1 = E0+self.rf_voltage*math.sin(self.rf_angular_frequency*time)
            p1 = (E1**2-self.mass**2)**0.5
            beta_rel = p1/E1
            new_time = time + self.circumference/beta_rel/xboa.common.constants["c_light"]
            self.dp_over_p_data[i] = p1/self.momentum - 1
            self.t_data[i] = new_time

    def foil_model(self, event_number):
        if not self.do_scattering and not self.do_energy_loss:
            return
        momentum = self.momentum*(1+self.dp_over_p_data[event_number])
        particle = Particle.new_from_momentum(momentum, self.pid)
        if self.do_scattering:
            sigma = self.foil.scattering(particle, self.foil_thickness) # thickness is in cm
            scat = numpy.random.randn(2)*sigma
            self.beam_data[event_number][1] += scat[0]
            self.beam_data[event_number][3] += scat[1]
            if self.verbose > 5 and event_number == 0:
                print("scattering sigma", sigma, "angles", scat, "psv", self.beam_data[event_number])
        if self.do_energy_loss:
            dE = self.foil.energy_loss_dz(particle)*self.foil_thickness #dE negative
            energy = particle.set_kinetic_energy(particle.get_kinetic_energy()+dE)
            dp_over_p = (particle.get_momentum()-self.momentum)/self.momentum
            if self.verbose > 5 and event_number == 0:
                print("Losing", dE, "of energy leaving", dp_over_p, "dp/p")
            self.dp_over_p_data[event_number] = dp_over_p

    def inside_foil(self, psv):
        theta = math.radians(self.foil_angle)
        dy = psv[2]*math.cos(theta)+psv[0]*math.sin(theta)
        return dy - self.foil_edge < 0

    def check_foil(self):
        self.foil_hits += [0]*(len(self.beam_data)-len(self.foil_hits))
        if self.verbose > 8:
            print("For foil", foil_dimensions)
        for i, hit in enumerate(self.beam_data):
            if self.verbose > 8:
                print("    Check foil", hit, end=" ")
            if self.inside_foil(hit):
                self.foil_hits[i] += 1
                self.foil_hit_positions.append((hit[0], hit[2]))
                self.foil_model(i)
                if self.verbose > 8:
                    print("hit")
            else:
                if self.verbose > 8:
                    print("miss")


############################## PLOTTING ########################

    def setup_subplots(self):
        self.fig = matplotlib.pyplot.figure(figsize=(20, 10))
        self.fig2 = matplotlib.pyplot.figure(figsize=(20, 10))
        self.fig3 = matplotlib.pyplot.figure(figsize=(20, 10))
        self.fig4 = matplotlib.pyplot.figure(figsize=(20, 10))

        self.axes4 = [
            self.fig4.add_subplot(1, 2, 1),
            self.fig4.add_subplot(2, 2, 2),
            self.fig4.add_subplot(2, 2, 4),
        ]
        self.axes2 = [
            self.fig2.add_subplot(2, 3, 1,  position=[0.06, 0.55, 0.26, 0.35]),
            self.fig2.add_subplot(2, 3, 3,  position=[0.38, 0.55, 0.26, 0.35]),
            self.fig2.add_subplot(2, 3, 5,  position=[0.70, 0.55, 0.26, 0.35]),
            self.fig2.add_subplot(2, 3, 2,  position=[0.06, 0.10, 0.26, 0.35]),
            self.fig2.add_subplot(2, 3, 4,  position=[0.38, 0.10, 0.26, 0.35]),
            self.fig2.add_subplot(2, 3, 6,  position=[0.70, 0.10, 0.26, 0.35]),
        ]
        self.axes = [
            self.fig.add_subplot(2, 3, 1,  position=[0.06, 0.55, 0.26, 0.35]),
            self.fig.add_subplot(2, 6, 7,  position=[0.06, 0.15, 0.10, 0.30]),
            self.fig.add_subplot(2, 6, 8,  position=[0.22, 0.15, 0.10, 0.30]),
            self.fig.add_subplot(2, 3, 2,  position=[0.38, 0.55, 0.26, 0.35]),
            self.fig.add_subplot(2, 6, 9,  position=[0.38, 0.10, 0.10, 0.35]),
            self.fig.add_subplot(2, 6, 10, position=[0.54, 0.10, 0.10, 0.35]),
            self.fig.add_subplot(2, 3, 3,  position=[0.70, 0.55, 0.26, 0.35]),
            self.fig.add_subplot(2, 6, 11, position=[0.70, 0.10, 0.10, 0.35]),
            self.fig.add_subplot(2, 6, 12, position=[0.86, 0.10, 0.10, 0.35]),
            self.fig.add_subplot(2, 7, 13,  position=[0.06, 0.10, 0.10, 0.05]),
            self.fig.add_subplot(2, 7, 14,  position=[0.22, 0.10, 0.10, 0.05]),
        ]
        #self.axes[9] = matplotlib.axes.Axes(self.fig, [0.06, 0.10, 0.10, 0.35])
        #self.axes[10] = matplotlib.axes.Axes(self.fig, [0.22, 0.10, 0.10, 0.35])
        #self.axes[10] = self.axes[2].twinx()

    def make_2d_hist(self, ax1, ax2, lab1, lab2, axes, hits, centre, range_):
        x_list = [hit[ax1] for hit in hits]
        y_list = [hit[ax2] for hit in hits]
        axes.clear()
        n_bins = int(max(len(x_list)**0.5/3, 10))
        axes.hist2d(x_list, y_list, n_bins)
        color = matplotlib.cm.get_cmap()(0)
        axes.set_facecolor(color)
        axes.set_xlim(-range_[ax1]+centre[ax1], range_[ax1]+centre[ax1])
        axes.set_ylim(-range_[ax2]+centre[ax2], range_[ax2]+centre[ax2])
        axes.set_xlabel(lab1, fontsize=self.f_size)
        axes.set_ylabel(lab2, fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)

    def make_a_plot(self, ax1, ax2, lab1, lab2, axes, hits, centre, range_):
        #add p1 off the edge of the plot; this forces the z axis
        p1 = [2*range_[ax1]+centre[ax1], 2*range_[ax2]+centre[ax2], 1.]
        const = [1, 1., 1, 1.]
        x_list = [p1[0]]+[hit[ax1]*const[ax1] for hit in hits]
        y_list = [p1[1]]+[hit[ax2]*const[ax2] for hit in hits]
        z_list = [1.]+[t/self.number_pulses for t in self.beam_injection_turn]
        colors = matplotlib.pyplot.cm.coolwarm
        if not self.accumulate:
            axes.clear()
        marker_size = 100/len(hits)**0.5
        axes.scatter(x_list, y_list, c=z_list, marker="o", cmap=colors, s=marker_size)
        axes.set_xlim(-range_[ax1]+centre[ax1], range_[ax1]+centre[ax1])
        axes.set_ylim(-range_[ax2]+centre[ax2], range_[ax2]+centre[ax2])
        axes.set_xlabel(lab1, fontsize=self.f_size)
        axes.set_ylabel(lab2, fontsize=self.f_size)

    def trajectory_plot(self, ax1, ax2, axes, trajectory, specials, color):
        const = [1, 1.0, 1, 1.0]
        x_list = [hit[ax1]*const[ax1] for hit in trajectory]
        y_list = [hit[ax2]*const[ax2] for hit in trajectory]
        axes.plot(x_list, y_list, color=color)
        x_list = [hit[ax1]*const[ax1] for hit in specials]
        y_list = [hit[ax2]*const[ax2] for hit in specials]
        axes.plot(x_list, y_list, 'o', fillstyle='none')

    def make_1d_plot(self, ax1, lab1, axes, hits, centre, range_):
        x_list = [hit[ax1] for hit in hits]
        n_bins = int(max(10, len(x_list)/10))
        axes.clear()
        axes.frameon = False
        #axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.hist(x_list, bins=n_bins)
        axes.set_xlim(-range_[ax1]+centre[ax1], range_[ax1]+centre[ax1])
        axes.set_xlabel(lab1, fontsize=self.f_size)
        #y_max = len(x_list)/n_bins*10
        #axes.set_ylim(0, y_max)

    def plot_bump_orbit(self):
        n = self.number_pulses
        turn = range(self.number_pulses)
        x_list, px_list, y_list, py_list = [None]*n, [None]*n, [None]*n, [None]*n
        u_list, pu_list, v_list, pv_list = [None]*n, [None]*n, [None]*n, [None]*n
        for i in turn:
            matrix = self.get_transfer_matrix(i)
            orbit = self.get_bump_orbit(i)
            x_list[i] = orbit[0]
            px_list[i] = orbit[1]
            y_list[i] = orbit[2]
            py_list[i] = orbit[3]
            decoupled_orbit = matrix.decoupled(orbit)
            u_list[i] = decoupled_orbit[0]
            pu_list[i] = decoupled_orbit[1]
            v_list[i] = decoupled_orbit[2]
            pv_list[i] = decoupled_orbit[3]
        figure = self.fig3
        axes = figure.add_subplot(2, 2, 1)
        axes.plot(turn, x_list, label="x [mm]")
        axes.plot(turn, y_list, label="y [mm]")
        axes.set_xlabel("Turn", fontsize=self.f_size)
        axes.set_ylabel("", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        axes.legend()
        axes = figure.add_subplot(2, 2, 2)
        axes.plot(turn, px_list, label="$p_x$ [MeV/c]")
        axes.plot(turn, py_list, label="$p_y$ [MeV/c]")
        axes.set_xlabel("Turn", fontsize=self.f_size)
        axes.set_ylabel("", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        axes.legend()
        axes = figure.add_subplot(2, 2, 3)
        axes.plot(turn, u_list, label="u")
        axes.plot(turn, v_list, label="v")
        axes.set_xlabel("Turn", fontsize=self.f_size)
        axes.set_ylabel("", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        axes.legend()
        axes = figure.add_subplot(2, 2, 4)
        axes.plot(turn, pu_list, label="$p_u$")
        axes.plot(turn, pv_list, label="$p_v$")
        axes.set_xlabel("Turn", fontsize=self.f_size)
        axes.set_ylabel("", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        axes.legend()
        figure.savefig(self.output_dir+"turn_position.png")

    def plot_spherical(self, ax1, ax2, ax3, lab1, lab2, lab3, rel_hits, aa_hits):
        tm = self.get_transfer_matrix()
        #rel_hits = self.transform_to_relative_coordinates(None, None)
        #aa_hits = [tm.coupled_to_nd_action_angles(hit) for hit in rel_hits]
        x_list = [math.degrees(hit[ax1]) for hit in aa_hits]
        y_list = [math.degrees(hit[ax2]) for hit in aa_hits]
        z_list = [math.degrees(hit[ax3]) for hit in aa_hits]
        colors = matplotlib.pyplot.cm.coolwarm
        axes = self.axes2[5]
        axes.clear()

        #marker_size = 100/len(aa_hits)**0.5
        #scat = axes.scatter(x_list, y_list, c=z_list, marker='o', cmap=colors, s=marker_size)
        #axes.set_xlabel(lab1, fontsize=self.f_size)
        #axes.set_ylabel(lab2, fontsize=self.f_size)
        #axes.tick_params(labelsize = self.l_size)
        aa_hits = self.convert_to_aa(rel_hits)

        u_list = [hit[1] for hit in aa_hits]
        v_list = [hit[3] for hit in aa_hits]
        r_list = [(hit[1]+hit[3]) for hit in aa_hits]
        axis_range =  [0., max(r_list)*1.1]
        axis_range_2d = [
            [-self.aa_range[1]+self.aa_centre[1], self.aa_range[1]+self.aa_centre[1]],
            [-self.aa_range[3]+self.aa_centre[3], self.aa_range[3]+self.aa_centre[3]],
        ]
        n_bins = int(max(20, len(r_list)/10))
        n_bins_2d = int(max(20, int(len(r_list)**0.5/5) ) )

        axes.hist2d(u_list, v_list, n_bins_2d, axis_range_2d)
        axes.set_xlabel("$A_x$ [mm]", fontsize=self.f_size)
        axes.set_ylabel("$A_y$ [mm]", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)

        axes = self.axes2[2]
        axes.clear()
        scat = axes.hist(r_list, n_bins, axis_range, label="$A_{4d}$")
        scat = axes.hist(u_list, n_bins, axis_range, histtype="step", label="$A_x$")
        scat = axes.hist(v_list, n_bins, axis_range, histtype="step", label="$A_y$")
        axes.set_xlabel("$A$ [mm]", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        axes.legend()

    def check_acceptance(self, aa_hits):
        self.in_transverse_acceptance = [hit[1] <= self.amplitude_acceptance and hit[3] <= self.amplitude_acceptance for hit in aa_hits]
        if self.rf_voltage == 0.0:
            self.in_longitudinal_acceptance = [True for hit in aa_hits]
            return
        t_data = [self.rf_bucket_contours.periodicarise(t, self.harmonic_number) for t in self.t_data]
        self.in_longitudinal_acceptance = [self.rf_bucket_contours.inside_acceptance(t_data[i], dp, False)  for i, dp in enumerate(self.dp_over_p_data)]

    def acceptance_count(self):
        outside_acceptance = [not self.in_transverse_acceptance[i] or not self.in_longitudinal_acceptance[i] for i in range(len(self.in_longitudinal_acceptance))]
        self.output["n_outside_acceptance"] = sum(outside_acceptance)
        outside_trans_acceptance = [not self.in_transverse_acceptance[i] for i in range(len(self.in_longitudinal_acceptance))]
        self.output["n_outside_acceptance_trans"] = sum(outside_trans_acceptance)
        outside_long_acceptance = [not self.in_longitudinal_acceptance[i] for i in range(len(self.in_longitudinal_acceptance))]
        self.output["n_outside_acceptance_long"] = sum(outside_long_acceptance)
        return self.output["n_outside_acceptance"]

    def amplitude_plots(self, aa_hits):
        #self.test_amplitude()
        #input("Press <CR> to continue")
        self.make_a_plot(1, 3, "$A_{u}$ [mm]", "$A_{v}$ [mm]", self.axes[6], aa_hits, self.aa_centre, self.aa_range)
        self.make_a_plot(0, 1, "$\\phi_{u}$ [deg]", "$A_{u}$ [mm]", self.axes[7], aa_hits, self.aa_centre, self.aa_range)
        self.make_a_plot(2, 3, "$\\phi_{v}$ [deg]", "$A_{v}$ [mm]", self.axes[8], aa_hits, self.aa_centre, self.aa_range)
        mean_au = numpy.mean([hit[1] for hit in aa_hits])
        mean_av = numpy.mean([hit[3] for hit in aa_hits])
        mean_a4 = numpy.mean([hit[1]+hit[3] for hit in aa_hits])
        if not self.accumulate: # in accumulate mode, subsequent turns write on top of each other and it becomes unreadable
            self.axes[7].text(0.05, 0.95, "$\\left<A_u\\right>/2$: "+format(mean_au/2, "6.4g"), transform=self.axes[7].transAxes)
            self.axes[8].text(0.05, 0.95, "$\\left<A_v\\right>/2$: "+format(mean_av/2, "6.4g"), transform=self.axes[8].transAxes)
            self.axes[6].text(0.05, 0.95, "$\\left<A_{4d}\\right>/4$: "+format(mean_a4/4., "6.4g"), transform=self.axes[6].transAxes)
        if self.verbose > 8:
            print("AA mean", numpy.mean(aa_hits, 0), "AA range", numpy.ptp(aa_hits, 0))

    def draw_foil(self, axes):
        dw = 100.0
        dt = self.foil_edge
        theta = math.radians(self.foil_angle)
        x = [dt*math.sin(theta)+dw*math.cos(theta), dt*math.sin(theta)-dw*math.cos(theta)]
        y = [dt*math.cos(theta)-dw*math.sin(theta), dt*math.cos(theta)+dw*math.sin(theta)]
        axes.plot(x, y, color="gray")

    def will_do_stats(self):
        if not self.do_stats:
            return False
        if not self.last_turn() and self.turn % self.stats_frequency != 0:
            return False
        return True

    def will_do_plots(self):
        if not self.do_plots:
            return False
        if not self.last_turn() and self.turn % self.plot_frequency != 0:
            return False
        return True

    def turn_summary(self):  
        if not self.will_do_stats() and not self.will_do_plots():
            return
        relative_hits = self.transform_to_relative_coordinates(copy.deepcopy(self.beam_data), None)
        decoupled_hits = self.convert_to_decoupled(relative_hits)
        aa_hits = self.convert_to_aa(relative_hits)
        if self.will_do_plots() or self.will_do_stats():
            self.check_acceptance(aa_hits)
        if self.will_do_plots():
            self.plot_one_turn(relative_hits, decoupled_hits, aa_hits)
        if self.will_do_stats():
            self.stat_one_turn(relative_hits, decoupled_hits, aa_hits)

    def stat_one_turn(self, relative_hits, decoupled_hits, aa_hits):
        mean_h = numpy.mean(self.foil_hits)
        self.output["mean_foil_hits"] = mean_h
        self.output["max_foil_hits"] = max(self.foil_hits)

        sigma_dp = numpy.std(self.dp_over_p_data)
        self.output["rms_dp_over_p"] = sigma_dp
        self.output["dp_over_p_1e-3"] = self.get_range(self.dp_over_p_data, 0.001, "r")
        self.output["dp_over_p_1e-2"] = self.get_range(self.dp_over_p_data, 0.01, "r")

        total_hits = sum(self.foil_hits)
        total_hits_prev = sum(self.output["hits_per_turn"])
        self.output["hits_per_turn"].append(total_hits-total_hits_prev)
        mean_au = numpy.mean([hit[1] for hit in aa_hits])
        mean_av = numpy.mean([hit[3] for hit in aa_hits])
        mean_a4 = numpy.mean([hit[1]+hit[3] for hit in aa_hits])
        self.output["rms_emittance_u"] = mean_au/2.
        self.output["rms_emittance_v"] = mean_av/2.
        self.output["rms_emittance_4d"] = mean_a4/4.
        self.output["amplitude_u_1e-3"] = self.get_range([hit[1] for hit in aa_hits], 0.001, "u")
        self.output["amplitude_v_1e-3"] = self.get_range([hit[3] for hit in aa_hits], 0.001, "u")
        self.output["amplitude_4d_1e-3"] = self.get_range([hit[1]+hit[3] for hit in aa_hits], 0.001, "u")
        self.output["amplitude_u_1e-2"] = self.get_range([hit[1] for hit in aa_hits], 0.01, "u")
        self.output["amplitude_v_1e-2"] = self.get_range([hit[3] for hit in aa_hits], 0.01, "u")
        self.output["amplitude_4d_1e-2"] = self.get_range([hit[1]+hit[3] for hit in aa_hits], 0.01, "u")
        self.output["amplitude_u_v_corr"] = numpy.corrcoef([[hit[1], hit[3]] for hit in aa_hits], rowvar=False).tolist()
        self.output["n_events"] = len(self.beam_data)
        self.output["beam_trajectory"] = [orbit.tolist() for orbit in self.bump_orbits]

        self.acceptance_count()


    def set_title(self, figure):
        title = "Turn "+str(self.turn)+"    Bumper setting "+str(self.get_bumper_index()+1)
        if self.injecting():
            title += "    Injecting"
        else:
            title += "    Moving trajectory"
        figure.suptitle(title)

    def plot_one_turn(self, relative_hits, decoupled_hits, aa_hits):
        plot_summary = True
        hits = copy.deepcopy(self.beam_data)
        real_range = self.real_range
        real_centre = self.real_centre

        dec_range = self.dec_range
        dec_centre = self.dec_centre

        self.set_title(self.fig)
        if self.verbose > 8:
            print("Real mean", numpy.mean(hits, 0), "Real range", numpy.ptp(hits, 0))
        co_trajectory = numpy.array([orbit for orbit in self.bump_orbits])
        co_specials = numpy.array([self.bump_orbits[0], self.bump_orbits[self.number_pulses], self.bump_orbits[-1]])
        bump_trajectory = numpy.array([orbit for orbit in self.injection_orbits])
        bump_specials = numpy.array([self.injection_orbits[0], self.injection_orbits[-1]])
        if plot_summary:
            self.make_2d_hist(0, 2, "x [mm]", "y [mm]", self.axes2[0], hits, real_centre, real_range)
            self.plot_foil_hits(self.real_range, self.real_centre)
        self.make_a_plot(0, 2, "x [mm]", "y [mm]", self.axes[0], hits, real_centre, real_range)
        distance, n_sigma = self.beam_from_foil_edge()
        self.axes[0].text(0.05, 0.95, "Distance to foil: "+format(distance, "4.2g")+" mm ("+format(n_sigma, "4.2g")+" s.d.)", transform=self.axes[0].transAxes)

        self.trajectory_plot(0, 2, self.axes[0], co_trajectory, co_specials, 'b')
        self.trajectory_plot(0, 2, self.axes[0], bump_trajectory, bump_specials, 'r')
        self.draw_foil(self.axes[0])
        self.make_a_plot(0, 1, "", "x'", self.axes[1], hits, real_centre, real_range)
        self.make_1d_plot(0, "x [mm]", self.axes[9], hits, real_centre, real_range)
        self.trajectory_plot(0, 1, self.axes[1], co_trajectory, co_specials, 'b')
        self.trajectory_plot(0, 1, self.axes[1], bump_trajectory, bump_specials, 'r')
        self.make_a_plot(2, 3, "", "y'", self.axes[2], hits, real_centre, real_range)
        self.make_1d_plot(2, "y [mm]", self.axes[10], hits, real_centre, real_range)
        self.trajectory_plot(2, 3, self.axes[2], co_trajectory, co_specials, 'b')
        self.trajectory_plot(2, 3, self.axes[2], bump_trajectory, bump_specials, 'r')


        inj_trajectory = [-numpy.array(orbit)+self.injection_orbits[0] for orbit in self.bump_orbits]
        inj_trajectory = self.convert_to_decoupled(inj_trajectory)
        inj_specials = [[0., 0., 0., 0.], -self.get_bump_orbit()+self.injection_orbits[0]]
        inj_specials = self.convert_to_decoupled(inj_specials)

        if self.verbose > 8:
            print("Decoupled mean", numpy.mean(decoupled_hits, 0), "Decoupled range", numpy.ptp(decoupled_hits, 0))
        self.make_a_plot(0, 2, "u", "v", self.axes[3], decoupled_hits, dec_centre, dec_range)
        self.trajectory_plot(0, 2, self.axes[3], inj_trajectory, inj_specials, 'g')
        tune_u = self.get_transfer_matrix().get_phase_advance(0)/math.pi/2.
        tune_v = self.get_transfer_matrix().get_phase_advance(1)/math.pi/2.
        self.make_a_plot(0, 1, "u", "u'", self.axes[4], decoupled_hits, dec_centre, dec_range)
        self.trajectory_plot(0, 1, self.axes[4], inj_trajectory, inj_specials, 'g')
        if not self.accumulate: # in accumulate mode, subsequent turns write on top of each other and it becomes unreadable
            self.axes[4].text(0.05, 0.95, "$\\nu_u$: "+format(tune_u, "6.4g"), transform=self.axes[4].transAxes)
        self.make_a_plot(2, 3, "v", "v'", self.axes[5], decoupled_hits, dec_centre, dec_range)
        self.trajectory_plot(2, 3, self.axes[5], inj_trajectory, inj_specials, 'g')
        if not self.accumulate: # in accumulate mode, subsequent turns write on top of each other and it becomes unreadable
            self.axes[5].text(0.05, 0.95, "$\\nu_v$: "+format(tune_v, "6.4g"), transform=self.axes[5].transAxes)

        self.acceptance_count()
        self.amplitude_plots(aa_hits)
        if plot_summary:
            self.plot_spherical(0, 1, 2, "$\\phi_1$ [deg]", "$\\phi_2$  [deg]", "$\\phi_3$  [deg]", relative_hits, aa_hits)

        name = "transverse_turn_"+str(self.turn).rjust(5, "0")+".png"
        self.fig.savefig(self.output_dir+name)
        if plot_summary:
            self.fig2.savefig(f"{self.output_dir}/summary_{str(self.turn).rjust(5, '0')}.png")

        if self.rf_voltage > 0.0:
            self.plot_longitudinal()
        if self.sleep_time > 0:
            matplotlib.pyplot.pause(self.sleep_time)

    def get_range(self, data, fraction, bound): # bound = (r)ange, (u)pper, (l)ower, (b)ounds
        n_events = len(data)
        dp_over_p_sorted = sorted(data)
        if bound == "r":
            n_delta = int(n_events*fraction/2)
            upper = dp_over_p_sorted[-1-n_delta]
            lower = dp_over_p_sorted[n_delta]
            return upper-lower
        elif bound == "l":
            n_delta = int(n_events*fraction)
            return dp_over_p_sorted[n_delta]
        elif bound == "u":
            n_delta = int(n_events*fraction)
            return dp_over_p_sorted[-1-n_delta]
        elif bound == "b":
            n_delta = int(n_events*fraction/2)
            bounds = [dp_over_p_sorted[n_delta], dp_over_p_sorted[-1-n_delta]]
            return bounds

    def build_bucket(self, will_replace):
        # note rf momentum may be different to nominal momentum
        if not will_replace and not self.rf_bucket_contours is None:
            return
        if self.rf_voltage == 0.0:
            return
        real_estate_gradient = self.rf_voltage/self.circumference
        frequency = self.rf_angular_frequency/2./math.pi
        rf_time = 1./frequency*self.harmonic_number/2.0 # time at centre of an rf bucket
        rf_mass = xboa.common.pdg_pid_to_mass[self.pid]
        rf_energy = (self.rf_reference_momentum**2+self.mass**2)**0.5
        t0 = 0.0
        t1 = self.ring_tof
        e0 = 0.0
        e1 = 1e9
        self.rf_bucket_contours = optimisation_tools.analysis.rf_bucket.RFBucket(
                            real_estate_gradient, frequency, rf_time, rf_energy,
                            self.mass, t0, t1, e0, e1)
        self.rf_bucket_contours.y_var = "dp_over_p"
        self.rf_bucket_contours.dz = self.circumference
        self.rf_bucket_contours.central_p = self.momentum # NOT the synchronous particle; rather the particle central to dp_over_p, which may be different
        self.rf_bucket_contours.pseudo_foil_de = self.foil_de_for_rf_bucket
        fp0, fp1 = self.rf_bucket_contours.fixed_points()
        for time in [fp0+1, fp1-1]:
            self.rf_bucket_contours.add_contour(time)
        acceptance = self.rf_bucket_contours.contours[-1]["area"]*0.75
        self.rf_bucket_contours.add_acceptance_contour(acceptance)

    def plot_longitudinal(self, highlight_acceptance=True):
        if not self.accumulate:
            for ax in self.axes4:
                ax.clear()
        marker_size = 10
        if len(self.dp_over_p_data) > 0:
            marker_size = 100/len(self.dp_over_p_data)**0.5
        colors = matplotlib.pyplot.cm.coolwarm

        axes = self.axes4[0]
        plot_time = self.harmonic_number*self.rf_angular_frequency/2.0/math.pi
        #t0_data = [t*plot_time for t in self.t_data]
        #t0_data = [(t - math.floor(t))/plot_time for t in t0_data]
        t0_data = [self.rf_bucket_contours.periodicarise(t, self.harmonic_number) for t in self.t_data]
        dp_data = self.dp_over_p_data
        self.rf_bucket_contours.plot_contours(axes)

        in_acceptance = [self.rf_bucket_contours.inside_acceptance(t0_data[i], dp, False) for i, dp in enumerate(dp_data)]
        t0_data_ii = [t0 for i, t0 in enumerate(t0_data) if self.in_longitudinal_acceptance[i] and self.in_transverse_acceptance[i]]
        dp_data_ii = [dp for i, dp in enumerate(dp_data) if self.in_longitudinal_acceptance[i] and self.in_transverse_acceptance[i]]
        t0_data_oi = [t0 for i, t0 in enumerate(t0_data) if not self.in_longitudinal_acceptance[i] and self.in_transverse_acceptance[i]]
        dp_data_oi = [dp for i, dp in enumerate(dp_data) if not self.in_longitudinal_acceptance[i] and self.in_transverse_acceptance[i]]
        t0_data_io = [t0 for i, t0 in enumerate(t0_data) if self.in_longitudinal_acceptance[i] and not self.in_transverse_acceptance[i]]
        dp_data_io = [dp for i, dp in enumerate(dp_data) if self.in_longitudinal_acceptance[i] and not self.in_transverse_acceptance[i]]
        t0_data_oo = [t0 for i, t0 in enumerate(t0_data) if not self.in_longitudinal_acceptance[i] and not self.in_transverse_acceptance[i]]
        dp_data_oo = [dp for i, dp in enumerate(dp_data) if not self.in_longitudinal_acceptance[i] and not self.in_transverse_acceptance[i]]

        my_count = "Number in longitudinal acceptance: "+str(len(t0_data_ii+t0_data_io))+"/"+str(len(self.in_longitudinal_acceptance))
        for (color, t_data, p_data) in [("blue", t0_data_ii+t0_data_io, dp_data_ii+dp_data_io),
                                        ("orange", t0_data_oi+t0_data_oo, dp_data_oi+dp_data_oo),
                                        ]:
            axes.scatter(t_data, p_data, c=color, marker="o", s=marker_size)
        axes.text(0.95, 0.95, my_count, transform=axes.transAxes, horizontalalignment='right')
        axes.set_xlabel("t [ns]", fontsize=self.f_size)
        axes.set_ylabel("dp/p", fontsize=self.f_size)
        axes.set_xlim([0.0, self.ring_tof])
        pmin, pmax = [self.rf_bucket_contours.pmin()*1.5, self.rf_bucket_contours.pmax()*1.5]
        ref_min = self.get_bump_orbit()+self.delta_co(pmin)
        ref_max = self.get_bump_orbit()+self.delta_co(pmax)

        axes.set_ylim([pmin, pmax])
        axes = self.axes4[1]
        z_list = [t/self.number_pulses for t in self.beam_injection_turn]
        x_list = [hit[0] for hit in self.beam_data]
        axes.scatter(x_list, self.dp_over_p_data, c=z_list, marker="o", cmap=colors, s=marker_size)
        axes.plot([self.get_bump_orbit()[0]], [0], 'o', fillstyle='none')
        axes.plot([ref_min[0], ref_max[0]], [pmin, pmax], '-')
        axes.set_xlabel("x [mm]", fontsize=self.f_size)
        axes.set_ylabel("dp/p", fontsize=self.f_size)
        axes.set_xlim([self.real_centre[0]-self.real_range[0], self.real_centre[0]+self.real_range[0]])
        axes.set_ylim([pmin, pmax])

        axes = self.axes4[2]
        x_list = [hit[2] for hit in self.beam_data]
        axes.scatter(x_list, self.dp_over_p_data, c=z_list, marker="o", cmap=colors, s=marker_size)
        axes.set_xlabel("y [mm]", fontsize=self.f_size)
        axes.plot([self.get_bump_orbit()[2]], [0], 'o', fillstyle='none')
        axes.plot([ref_min[2], ref_max[2]], [pmin, pmax], '-')
        axes.set_ylabel("dp/p", fontsize=self.f_size)
        axes.set_xlim([self.real_centre[2]-self.real_range[2], self.real_centre[2]+self.real_range[2]])
        axes.set_ylim([pmin, pmax])

        name = "longitudinal_turn_"+str(self.turn).rjust(5, "0")+".png"
        self.set_title(self.fig4)
        self.fig4.savefig(os.path.join(self.output_dir, name))

    def plot_foil_hits(self, range_, centre):
        n_bins = 50 #max(self.foil_hits)+2
        axes = self.axes2[1]
        axes.clear()
        axes.hist(self.foil_hits, n_bins, (-0.5, n_bins-0.5), rwidth=0.2)
        axes.set_xlabel("Number of foil hits", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        mean_h = numpy.mean(self.foil_hits)
        axes.text(0.05, 0.95, "<hits> "+format(mean_h, "6.4g"), transform=axes.transAxes)

        axes = self.axes2[4]
        axes.clear()
        axes.hist(self.dp_over_p_data, n_bins, (-0.02, 0.02))
        axes.set_xlabel("dp/p", fontsize=self.f_size)
        axes.tick_params(labelsize = self.l_size)
        sigma_dp = numpy.std(self.dp_over_p_data)
        bounds_99 = self.get_range(self.dp_over_p_data, 0.01, "b") # 99% contour
        axes.text(0.05, 0.95, "$\\sigma(dp/p)$ "+format(sigma_dp, "6.4g"), transform=axes.transAxes)
        axes.text(0.05, 0.90, "99% dp/p "+format(bounds_99[1]-bounds_99[0], "6.4g"), transform=axes.transAxes)
        ylim = axes.get_ylim()
        axes.plot([bounds_99[0]]*2, ylim, c="grey", linestyle="dashed")
        axes.plot([bounds_99[1]]*2, ylim, c="grey", linestyle="dashed")
        axes.set_ylim(ylim)

        axes = self.axes2[3]
        axes.clear()
        x_list = [pos[0] for pos in self.foil_hit_positions]
        y_list = [pos[1] for pos in self.foil_hit_positions]
        marker_size = 10
        if len(self.foil_hit_positions) > 0:
            marker_size = 100/len(self.foil_hit_positions)**0.5
        axes.scatter(x_list, y_list, marker="o", s=marker_size)

        x_list = [pos[0] for pos in self.first_turn_positions]
        y_list = [pos[1] for pos in self.first_turn_positions]
        axes.scatter(x_list, y_list, marker="o", s=marker_size)

        self.draw_foil(axes)

        axes.set_xlim(-range_[0]+centre[0], range_[0]+centre[0])
        axes.set_ylim(-range_[2]+centre[2], range_[2]+centre[2])
        axes.set_xlabel("x [mm]", fontsize=self.f_size)
        axes.set_ylabel("y [mm]", fontsize=self.f_size)


    def get_output_parameters(self, parameter_list_of_dicts = None):
        if parameter_list_of_dicts == None:
            parameter_list_of_dicts = []
        output = copy.deepcopy(self.output)
        for a_dict in parameter_list_of_dicts:
            for key, value in a_dict.items():
                if key in self.__dict__:
                    output[key] = self.__dict__[key]
                else:
                    output[key] = value
        return output

    def print_output(self):
        for key, value in self.output.items():
            try:
                print(key, format(value, "6.4g"))
            except TypeError:
                print(key, value)
        print()

    def movie(self):
        here = os.getcwd()
        os.chdir(self.output_dir)
        #mencoder mf://turn*.png -mf w=800:h=600:fps=5:type=png -ovc lavc -lavcopts vcodec=msmpeg4:mbd=2:trell -oac copy -o injection.avi
        try:
            output = subprocess.check_output(["mencoder",
                                    "mf://transverse_turn_*.png",
                                    "-mf", "w=1024:h=600:fps=5:type=png",
                                    "-ovc", "lavc",
                                    "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                    "-oac", "copy",
                                    "-o", "transverse_injection.avi"])
        except:
            print("Movie failed")
        try:
            output = subprocess.check_output(["mencoder",
                                    "mf://longitudinal_turn_*.png",
                                    "-mf", "w=1024:h=600:fps=5:type=png",
                                    "-ovc", "lavc",
                                    "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                    "-oac", "copy",
                                    "-o", "longitudinal_injection.avi"])
        except:
            print("bob")
        try:
            output = subprocess.check_output(["mencoder",
                                    "mf://summary_0*.png",
                                    "-mf", "w=1024:h=600:fps=5:type=png",
                                    "-ovc", "lavc",
                                    "-lavcopts", "vcodec=msmpeg4:vbitrate=2000:mbd=2:trell",
                                    "-oac", "copy",
                                    "-o", "summary.avi"])
        except:
            print("bob")
        os.chdir(here)

    def finalise(self, will_clear = True):
        if self.beam_filename:
            self.print_beam(self.output_dir+self.beam_filename)
        if not self.do_plots:
            return
        self.plot_foil_hits(self.real_range, self.real_centre)
        self.plot_bump_orbit()
        if self.max_turn/self.plot_frequency > 1 and self.do_movie:
            self.movie()
        self.fig.savefig(self.output_dir+"phase_space_final.png")
        self.fig2.savefig(self.output_dir+"summary_final.png")
        if will_clear:
            self.fig.clear()
            matplotlib.pyplot.close(self.fig)
            self.fig2.clear()
            matplotlib.pyplot.close(self.fig2)
            self.fig3.clear()
            matplotlib.pyplot.close(self.fig3)
            self.fig4.clear()
            matplotlib.pyplot.close(self.fig4)

    amp_scale = 3.0 # scaling for axes
