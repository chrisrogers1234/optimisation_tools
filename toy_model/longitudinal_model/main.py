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
import analysis.monitor_analysis
import instrumentation.monitor_digitisation
import beam.beam_factory
import beam.particle
import transport.turn_action
import transport.longitudinal_model
import transport.ffa_model
import transport.synchrotron_model
import rf_programme.piecewise_interpolation

def main_voltage_ramp(do_tracking, do_analysis):
    model = transport.synchrotron_model.SynchrotronModel()
    model.harmonic_number = 2

    energy = 70
    p_start = beam.particle.Particle(0, energy, beam.beam_factory.BeamFactory.mass)
    p_mid = beam.particle.Particle(0, energy, beam.beam_factory.BeamFactory.mass)
    p_start.set_momentum(p_mid.momentum()*(1-0.0015))
    p_end = beam.particle.Particle(0, energy, beam.beam_factory.BeamFactory.mass)
    p_end.set_momentum(p_mid.momentum()*(1+0.0015))
    t0 = model.get_time_of_flight(p_start)
    tr = model.get_time_of_flight(p_mid)
    t1 = model.get_time_of_flight(p_end)
    print(f"Setting up lattice for energy {energy}, t0 {tr}")
    print(f"    Beam momenta from {p_start.momentum()} to {p_end.momentum()}")
    print(f"         energy from {p_start.energy} to {p_end.energy}")
    p_list = [p_mid]+beam.beam_factory.BeamFactory.make_coasting_beam_square(10000, p_start.energy, p_end.energy, n_turns=100, model=model)
    ramp_time = 2*1e6
    ramp_time = 4*1e6
    program = rf_programme.piecewise_interpolation.PiecewiseInterpolation()
    program.v_list = [0.0002, 0.0002,            0.004,              0.006,               0.006]
    program.t_list = [0,      t1*200, t1*200+ramp_time, t1*200+ramp_time*2,  t1*200+ramp_time*3]
    program.f_list = [1.0/tr*model.harmonic_number]*len(program.v_list)
    max_time = program.t_list[-1]
    program.setup(max_time)
    model.rf_program = program
    output_directory = f"output/voltage_bump_v13"
    monitor = instrumentation.monitor_digitisation.MonitorDigitisation()
    monitor.file_name = os.path.join(output_directory, "monitor.dat")
    monitor.t_resolution = 10 # ns

    turn_action = transport.turn_action.TurnAction(program, monitor, model, plot_contours=True)
    turn_action.plot_frequency = 10
    turn_action.output_directory = output_directory
    model.do_turn_action = turn_action.do_turn_action
    if do_tracking:
        utilities.clear_dir(output_directory)
        model.track_beam(max_time = max_time, max_turn = None, particle_collection=p_list)
        monitor.save()
    print("Done tracking - finishing up")
    model.write_rf_data(output_directory+"/rf.dat", 10.0, max_time)

    if do_analysis:
        my_analysis = analysis.monitor_analysis.MonitorAnalysis()
        my_analysis.model = model
        my_analysis.monitor = monitor
        my_analysis.output_directory = output_directory
        monitor.load()
        my_analysis.do_plot()

if __name__ == "__main__":
    main_voltage_ramp(True, True)
    #matplotlib.pyplot.show(block=False)
    #input("Press <CR> to finish")
    #test_monitor()
    """
    for i in range(6):
        config = (57+i*0.1, 0.1, 0.004, 56, 4540, 4540, 2000, f"constant_bucket_central_energy_{i}")
        main_fork(config)
    for i in range(6):
        config = (57+i*0.1, 0.001, 0.004, 56, 4540, 4540, 2000, f"constant_bucket_central_energy_no_e_spread_{i}")
        main_fork(config)
    """
