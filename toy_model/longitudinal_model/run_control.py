import json
import os
import copy
import math
import bisect

import scipy.interpolate
import scipy.integrate
import numpy
import numpy.random

import matplotlib
import matplotlib.pyplot

import optimisation_tools.utils.utilities as utilities

import optimisation_tools.toy_model.longitudinal_model.beam as beam
import optimisation_tools.toy_model.longitudinal_model.transport as transport
import optimisation_tools.toy_model.longitudinal_model.instrumentation as instrumentation
import optimisation_tools.toy_model.longitudinal_model.analysis as analysis
import optimisation_tools.toy_model.longitudinal_model.rf_programme as rf_programme
import optimisation_tools.toy_model.longitudinal_model.real_data as real_data
import real_data.extract
import real_data.extract_rf_data
import real_data.extract_ct_data
import analysis.monitor_analysis
import analysis.postprocessing_output
import instrumentation.monitor_digitisation
import beam.beam_factory
import beam.particle
import transport.turn_action
import transport.longitudinal_model
import transport.ffa_model
import transport.synchrotron_model
import rf_programme.piecewise_interpolation

class RunControl:
    """
    Convert from a json configuration file and run the simulation

    Handles all of the low level conversion parameters/units/offsets
    """
    def __init__(self):
        self.config = None
        self.model = None
        self.p_list = []
        self.monitor = None

    def setup(self, config):
        self.config = config
        self.setup_model()
        self.setup_beam()
        self.setup_rf_programme()
        self.setup_monitor()
        self.setup_turn_action()

    def run(self):
        output_dir = self.config["run_control"]["output_directory"]
        print(f"Running longitudinal things in directory {output_dir}")
        if self.config["run_control"]["clear_dir"]:
            utilities.clear_dir(output_dir)
        self.dump_config()
        self.run_tracking()
        self.run_mc_analysis()
        self.run_recon_analysis()
        print(f"Done run in directory {output_dir}")

    def setup_model(self):
        plugin = self.config["model"]["plugin"]
        if plugin["type"] == "synchrotron":
            self.model = transport.synchrotron_model.SynchrotronModel()
            emin = plugin["energy_min"]
            emax = plugin["energy_max"]
            mass = self.model.mass
            self.model.momentum_min = ((emin+mass)**2-mass**2)**0.5
            self.model.momentum_max = ((emax+mass)**2-mass**2)**0.5
            self.model.magnet_period = plugin["magnet_period"]
            self.model.magnet_minimum = plugin["magnet_minimum"]
            self.model.phase_slip = plugin["phase_slip"]
        self.model.harmonic_number = self.config["model"]["harmonic_number"]
        self.model.r0 = self.config["model"]["r0"]

    def setup_beam(self):
        energy = self.config["beam"]["energy"]
        dp_over_p = self.config["beam"]["dp_over_p"]
        n_p = self.config["beam"]["n_particles"]
        n_injection_turns = self.config["beam"]["n_injection_turns"]
        p_start = beam.particle.Particle(0, energy, beam.beam_factory.BeamFactory.mass)
        p_mid = beam.particle.Particle(0, energy, beam.beam_factory.BeamFactory.mass)
        p_start.set_momentum(p_mid.momentum()*(1-dp_over_p))
        p_end = beam.particle.Particle(0, energy, beam.beam_factory.BeamFactory.mass)
        p_end.set_momentum(p_mid.momentum()*(1+dp_over_p))
        p_list = [p_mid]+beam.beam_factory.BeamFactory.make_coasting_beam_square(
                n_p, p_start.energy, p_end.energy,
                n_turns=n_injection_turns, model=self.model)
        self.p_list = p_list

    def setup_rf_programme(self):
        programme = rf_programme.piecewise_interpolation.PiecewiseInterpolation()
        dt = self.config["rf"]["time_delay"]
        programme.v_list = [rf["voltage"] for rf in self.config["rf"]["programme"]]
        programme.t_list = [rf["time"]+dt for rf in self.config["rf"]["programme"]]
        programme.f_list = [rf["frequency"] for rf in self.config["rf"]["programme"]]
        programme.setup(self.config["run_control"]["max_time"])
        self.model.rf_program = programme

    def setup_monitor(self):
        monitor = instrumentation.monitor_digitisation.MonitorDigitisation()
        file_name = self.config["mc_monitor"]["file_name"]
        monitor.file_name = os.path.join(self.config["run_control"]["output_directory"], file_name)
        monitor.t_resolution = self.config["mc_monitor"]["time_step"]
        self.monitor = monitor

    def setup_turn_action(self):
        contours = self.config["turn_action"]["plot_contours"]
        turn_action = transport.turn_action.TurnAction(self.model.rf_program, self.monitor, self.model, plot_contours=contours)
        turn_action.plot_frequency = self.config["turn_action"]["plot_frequency"]
        turn_action.output_directory = self.config["run_control"]["output_directory"]
        self.model.do_turn_action = turn_action.do_turn_action
        self.turn_action = turn_action

    def dump_config(self):
        output_dir = self.config["run_control"]["output_directory"]
        config_filename = self.config["run_control"]["config_filename"]
        with open(os.path.join(output_dir, config_filename), "w") as fout:
            fout.write(json.dumps(self.config, indent=2))

    def run_tracking(self):
        if not self.config["run_control"]["do_tracking"]:
            return
        output_dir = self.config["run_control"]["output_directory"]
        n_turns = self.config["run_control"]["max_time"]*self.config["rf"]["programme"][-1]["frequency"]/self.config["model"]["harmonic_number"]
        print("approx number of turns", n_turns)
        print("Model track beam...")
        self.model.track_beam(max_time = self.config["run_control"]["max_time"],
                              max_turn = self.config["run_control"]["max_turn"],
                              particle_collection=self.p_list)
        print("Save monitor")
        self.monitor.save()
        file_name = os.path.join(output_dir, self.config["rf"]["file_name"])
        t_step = self.config["rf"]["time_step"]
        print("Save rf data")
        self.model.write_rf_data(file_name, t_step, self.config["run_control"]["max_time"])

    def run_mc_analysis(self):
        if not self.config["run_control"]["do_mc_analysis"]:
            return
        file_prefix = self.config["mc_monitor"]["plot_prefix"]
        print("Load monitor")
        self.monitor.load()
        self.run_analysis(self.monitor, file_prefix)

    def setup_rf_data(self):
        rf_data = real_data.extract_rf_data.ExtractRFData()
        rf_data.file_glob = self.config["real_rf"]["file_glob"]
        rf_data.output_directory = self.config["run_control"]["output_directory"]
        rf_data.problem_title = self.config["run_control"]["problem_title"]
        rf_data.load_data()
        rf_data.extract_data()
        return rf_data

    def setup_ct_data(self):
        ct_data = real_data.extract_ct_data.ExtractCTData()
        ct_data.file_glob = self.config["real_ct"]["file_glob"]
        ct_data.output_directory = self.config["run_control"]["output_directory"]
        ct_data.problem_title = self.config["run_control"]["problem_title"]
        ct_data.load_data()
        ct_data.extract_data()
        return ct_data

    def setup_monitor_data(self):
        monitor = real_data.extract.ExtractRealData()
        monitor.time_key = self.config["real_monitor"]["time_key"]
        monitor.data_key = self.config["real_monitor"]["data_key"]
        monitor.stroke = self.config["real_monitor"]["stroke"]
        monitor.file_glob = self.config["real_monitor"]["file_glob"]
        monitor.time_units = self.config["real_monitor"]["time_units"]
        monitor.load_data()
        monitor.extract_data()
        return monitor

    def run_recon_analysis(self):
        if not self.config["run_control"]["do_real_analysis"]:
            return
        ct_data = self.setup_ct_data()
        rf_data = self.setup_rf_data()
        monitor = self.setup_monitor_data()
        file_prefix = self.config["real_monitor"]["plot_prefix"]
        my_analysis = self.run_analysis(monitor, ct_data, rf_data, file_prefix)
        self.run_postprocessing_output(my_analysis, rf_data)

    def run_analysis(self, monitor, ct_data, rf_data, prefix):
        my_analysis = analysis.monitor_analysis.MonitorAnalysis()
        my_analysis.model = self.model
        my_analysis.monitor = monitor
        my_analysis.ct = ct_data
        my_analysis.rf_data = rf_data
        my_analysis.config = self.config
        my_analysis.problem_title = self.config["run_control"]["problem_title"]
        my_analysis.max_oned_histo_size = self.config["analysis"]["max_oned_histo_size"]
        my_analysis.output_directory = self.config["run_control"]["output_directory"]
        my_analysis.file_prefix = prefix
        print("Do plot")
        my_analysis.do_plot()
        return my_analysis

    def run_postprocessing_output(self, my_analysis, rf_data):
        pp_output = analysis.postprocessing_output.PostprocessingOutput()
        pp_output.output_filename = f'{self.config["run_control"]["output_directory"]}/{self.config["run_control"]["postprocessing_filename"]}'
        pp_output.rf_data = rf_data
        pp_output.my_analysis = my_analysis
        pp_output.config = self.config
        pp_output.process()
