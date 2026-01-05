import math

import optimisation_tools.utils.utilities as utilities


import optimisation_tools.toy_model.longitudinal_model.beam as beam
import optimisation_tools.toy_model.longitudinal_model.transport as transport
import optimisation_tools.toy_model.longitudinal_model.instrumentation as instrumentation
import optimisation_tools.toy_model.longitudinal_model.analysis as analysis
import beam.beam_factory

class LongitudinalModel(object):
    def __init__(self):
        self.r0 = 26000 # mm
        self.phi0 = 0.0 # rad
        self.mass = beam.beam_factory.BeamFactory.mass # MeV/c^2
        self.c_light = 299.792458 # mm/ns
        self.harmonic_number = 1
        self.rf_program = None

    def path_length(self, a_particle):
        """Return path length in mm"""
        raise NotImplementedError("Path length not implemented")

    def get_rf_energy_change(self, a_particle):
        dE = self.rf_program.get_voltage(a_particle.t)
        #BUG is that we need to do some pseudo flooring before calculating the voltage.
        #phase = f0*a_particle.t-math.floor(f0*a_particle.t)
        #print(format(v0, "8.6g"), format(f0, "8.6g"), format(a_particle.t, "8.6g"), format(phase, "8.6g"), format(dE, "8.6g"), a_particle.energy)
        return dE

    def get_time_of_flight(self, a_particle):
        raise NotImplementedError("Path length not implemented")

    def _get_reference_energy(self, t, energy_estimate):
        tof = 1./self.rf_program.get_frequency(t)*self.harmonic_number
        beta = self.path_length(beam.particle.Particle(t, energy_estimate, self.mass))/tof/self.c_light # BUG
        energy = self.mass * 1.0/(1.0-beta**2)**0.5
        return energy-self.mass

    def get_reference_energy(self, t, energy_estimate, tolerance):
        """
        For a given time-of-flight, find the energy
        """
        delta_energy = energy_estimate
        while delta_energy > tolerance:
            new_energy_estimate = self._get_reference_energy(t, energy_estimate)
            delta_energy = abs(new_energy_estimate - energy_estimate)
            energy_estimate = new_energy_estimate
        return energy_estimate

    def one_turn(self, a_particle):
        dt = self.get_time_of_flight(a_particle)
        a_particle.t += dt
        dE = self.get_rf_energy_change(a_particle)
        a_particle.energy += dE
        return a_particle

    def get_ref_delta_energy(self, test_particle):
        f0 = self.rf_program.get_frequency(test_particle.t)
        e0 = self.get_reference_energy(test_particle.t, test_particle.energy, 1e-12)
        e1 = self.get_reference_energy(test_particle.t+1/f0, test_particle.energy, 1e-12)
        return e1-e0

    def set_to_reference_particle(self, a_particle):
        """Move the particle to the reference point, in the same rf period"""
        dE = self.get_ref_delta_energy(a_particle)
        dt = self.rf_program.get_relative_time(a_particle.t) # relative to 0 crossing
        f0 = self.rf_program.get_frequency(a_particle.t)
        v0 = self.rf_program.get_voltage_magnitude(a_particle.t)
        if math.fabs(v0) > 0.0 and math.fabs(dE/v0) < 1.0:
            reference_phase = math.asin(dE/v0)
        else:
            reference_phase = 0.0
        actual_phase = dt*f0*2*math.pi
        delta_t = (actual_phase-reference_phase)/2/math.pi/f0
        if delta_t < -1/f0/2:
            delta_t += 1/f0
        if delta_t > 1/f0/2:
            delta_t -= 1/f0
        old_time = a_particle.t
        new_time = a_particle.t - delta_t
        a_particle.t = new_time
        return
        print("Setting reference particle: Reference dE", dE,
              "old delta e", self.rf_program.get_voltage(old_time),
              "ref phase required", reference_phase,
              "ref phase now", actual_phase,
              "ref time now", old_time,
              "rf period", 1/f0,
              "new_time", new_time,
              "new delta e", self.rf_program.get_voltage(new_time))

    def one_turn_beam(self, a_particle_collection):
        for a_particle in a_particle_collection:
            self.one_turn(a_particle)

    def track_beam(self, max_time, max_turn, particle_collection):
        if (not max_time) and (not max_turn):
            raise ValueError("Must specify a max turn or max time")
        turn = 0
        while True:
            self.one_turn_beam(particle_collection)
            if max_time:
                particle_collection = [p for p in particle_collection if p.t <= max_time]
            particle_collection = [p for p in particle_collection if p.energy > 0.0]
            self.do_turn_action(turn, particle_collection)
            turn += 1
            if max_turn and turn > max_turn:
                break
            if len(particle_collection) == 0:
                break
            self.set_to_reference_particle(particle_collection[0])

    def do_turn_action(self, turn, particle_collection):
        pass

    def write_rf_data(self, file_name, time_step, max_time):
        fout = open(file_name, "w")
        t = 0
        while t < max_time:
            pseudo_particle = beam.particle.Particle(t, 0.0, 0.0)
            v = self.get_rf_energy_change(pseudo_particle)
            fout.write(f"{t} {v}\n")
            t += time_step

    def plot_rf_data(self, axes, n_points):
        axes_twin = axes.twinx()
        xlim = axes.get_xlim()
        x_list = []
        step = (xlim[1]-xlim[0])/(n_points-1)
        t_list = [xlim[0]+step*i for i in range(n_points)]
        f_list = [self.rf_program.get_frequency(t) for t in t_list]
        v_list = [self.rf_program.get_voltage(t) for t in t_list]
        axes_twin.plot(t_list, f_list, color="grey", linestyle="--")
        axes_twin.set_ylabel("f [GHz]")
        axes_twin.set_xlim(xlim)
