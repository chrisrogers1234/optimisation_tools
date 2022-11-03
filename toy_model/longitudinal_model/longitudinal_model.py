import os
import copy
import math
import bisect

import scipy.interpolate
import numpy
import numpy.random

import matplotlib
import matplotlib.pyplot

import optimisation_tools.utils.utilities as utilities

class LongitudinalModel(object):
    def __init__(self):
        self.r0 = 4540 # mm
        self.k = 7.6
        self.phi0 = 0.0 # rad
        self.mass = 938.272 # MeV/c^2
        self.c_light = 300 # mm/ns
        self.rf_program = ConstantBucket()

    def path_length(self, a_particle):
        """Return path length in mm"""
        p0 = ((11+938.272)**2-938.272**2)**0.5 # injection momentum
        p1 = a_particle.momentum()
        r1 = self.r0*(p1/p0)**(1/self.k)
        return r1*2*math.pi

    def get_rf_energy_change(self, a_particle):
        dE = self.rf_program.get_voltage(a_particle.t)
        #BUG is that we need to do some pseudo flooring before calculating the voltage.
        #phase = f0*a_particle.t-math.floor(f0*a_particle.t)
        #print(format(v0, "8.6g"), format(f0, "8.6g"), format(a_particle.t, "8.6g"), format(phase, "8.6g"), format(dE, "8.6g"), a_particle.energy)
        return dE

    def get_time_of_flight(self, a_particle):
        beta = a_particle.beta()
        dt = self.path_length(a_particle)/beta/self.c_light
        return dt

    def _get_reference_energy(self, t, harmonic_number, mass, energy_estimate):
        tof = 1./self.rf_program.get_frequency(t)*harmonic_number
        beta = self.path_length(Particle(t, energy_estimate, mass))/tof/self.c_light # BUG
        energy = mass * 1.0/(1.0-beta**2)**0.5
        return energy-mass

    def get_reference_energy(self, t, harmonic_number, mass, energy_estimate, tolerance):
        delta_energy = energy_estimate
        while delta_energy > tolerance:
            new_energy_estimate = self._get_reference_energy(t, harmonic_number, mass, energy_estimate)
            delta_energy = abs(new_energy_estimate - energy_estimate)
            energy_estimate = new_energy_estimate
        return energy_estimate

    def one_turn(self, a_particle):
        dt = self.get_time_of_flight(a_particle)
        a_particle.t += dt
        dE = self.get_rf_energy_change(a_particle)
        a_particle.energy += dE
        #print(dt, dE)
        return a_particle

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
            self.do_turn_action(turn, particle_collection)
            turn += 1
            if max_turn and turn > max_turn:
                break
            if len(particle_collection) == 0:
                break

    def do_turn_action(self, turn, particle_collection):
        pass

    def write_rf_data(self, file_name, time_step, max_time):
        fout = open(file_name, "w")
        t = 0
        while t < max_time:
            pseudo_particle = Particle(t, 0.0, 0.0)
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
        #print(t_list, "\n", f_list, "\n", v_list, "\n")
        axes_twin.plot(t_list, f_list, color="grey", linestyle="--")
        axes_twin.set_ylabel("f [GHz]")
        axes_twin.set_xlim(xlim)


class BeamMonitor(object):
    def __init__(self):
        self.t_list = []
        self.e_list = []

    def do_one_turn(self, turn, particle_collection):
        for p in particle_collection:
            self.t_list.append(p.t)
            self.e_list.append(p.energy)

    def do_plot(self, t_resolution, output_directory, model):
        dt = 50000
        t0, t1 = min(self.t_list), max(self.t_list)
        n_bins = int((t1-t0)/t_resolution)+1

        figure = matplotlib.pyplot.figure(figsize=(20,10))
        axes = figure.add_subplot(1, 1, 1)
        binned_data, bin_edges, patches = axes.hist(self.t_list, bins=n_bins)
        axes.set_title("Beam Monitor")
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("N")
        model.plot_rf_data(axes, 100)
        figure.savefig(os.path.join(output_directory, "monitor.png"))
        with open(os.path.join(output_directory, "monitor.dat"), "w") as fout:
            for i in range(n_bins):
                fout.write(f"{bin_edges[i]} {binned_data[i]}\n")
        tmin, tmax = t0, dt
        index = 0
        while tmax < t1:
            axes.set_xlim(tmin, tmax)
            figure.savefig(os.path.join(output_directory, "monitor_"+str(index)+".png"))
            tmin += dt
            tmax += dt
            index += 1
        return figure

class RFProgram(object):
    def __init__(self):
        pass

    def get_frequency(self, t):
        raise NotImplementedError()

    def get_voltage(self, t):
        raise NotImplementedError()

    def get_frequency_list(self, t_list):
        return [self.get_frequency(t) for t in t_list]

    def get_voltage_list(self, t_list):
        return [self.get_voltage(t) for t in t_list]

class ConstantBucket(object):
    def __init__(self):
        self.v0 = 0.004
        self.f0 = 1./393.0

    def get_frequency(self, t):
        return self.f0

    def get_voltage(self, t):
        return self.v0

class LinearFrequencyRamp(object):
    def __init__(self):
        self.v0 = 0.0026
        self.f0 = 0.0024
        self.f1 = 0.004
        self.t0 = 0
        self.t1 = 4e5

    def get_frequency(self, t):
        return (self.f1-self.f0)/(self.t1-self.t0)*(t-self.t0) + self.f0

    def get_voltage(self, t):
        return self.v0

class PiecewiseInterpolation(object):
    def __init__(self):
        self.v_list = []
        self.f_list = []
        self.t_list = []
        self.lookup_time = []
        self.k = 1

    def setup(self, max_time):
        self.v_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.v_list, k=self.k, s=0)
        self.f_interpolation = scipy.interpolate.UnivariateSpline(self.t_list, self.f_list, k=self.k, s=0)
        t = 0.0
        self.lookup_time = [t]
        while t < max_time:
            t = self.lookup_time[-1]
            freq = float(self.f_interpolation(t))
            self.lookup_time.append(t+1./freq)

    def get_voltage(self, t):
        index = bisect.bisect_left(self.lookup_time, t)
        try:
            dt = t-self.lookup_time[index]
        except IndexError:
            dt = t-self.lookup_time[-1]
        freq = self.f_interpolation(t)
        v0 = self.v_interpolation(t)
        volts = v0*math.sin(2*math.pi*freq*dt)
        return volts

    def get_frequency(self, t):
        return self.f_interpolation(t)


    #def get_frequency(self, t):
    #    return float(self.f_interpolation(t))
#
    #def get_voltage(self, t):
    #    return float(self.v_interpolation(t))

class Particle(object):
    def __init__(self, t, energy, mass):
        self.t = t
        self.energy = energy
        self.mass = mass

    def momentum(self):
        total_energy = self.energy + self.mass
        momentum = (total_energy**2-self.mass**2)**0.5
        return momentum

    def beta(self):
        total_energy = self.energy + self.mass
        momentum = (total_energy**2-self.mass**2)**0.5
        beta = momentum/total_energy
        return beta

class BeamFactory(object):
    def __init__(self):
        pass

    @classmethod
    def make_coasting_beam_square(cls, n_events, low_energy, high_energy, model):
        tof_list = []
        e_list = numpy.random.uniform(low_energy, high_energy, n_events)
        p_list = [Particle(0, e, cls.mass) for e in e_list]
        for p in p_list:
            tof = model.get_time_of_flight(p)
            p.t = numpy.random.uniform(0.0, tof)
            tof_list.append(tof)
        print("f:", 1./min(tof_list), 1./max(tof_list))
        return p_list

    @classmethod
    def make_uniform_distribution(cls, n_events, low_time, high_time, low_energy, high_energy):
        p_list= []
        for i in range(n_events):
            p_list.append(cls.make_p(numpy.random.uniform(low_time, high_time), numpy.random.uniform(low_energy, high_energy)))
        return p_list

    @classmethod
    def make_p(cls, t, energy):
        p = Particle(t, energy, BeamFactory.mass)
        return p

    mass = 938.272

class PlotBeam(object):
    def __init__(self, output_directory):
        self.t_bins = [i*360.0/100 for i in range(0, 100+1)]
        self.e_range = [19.0, 23.0]
        self.s = None
        self.output_directory = output_directory

    def plot_beam(self, p_list, model, plot_contours, suffix):
        if len(p_list) == 0:
            return
        t_list = [self.t_norm(p, p_list[0], model) for p in p_list]
        e_list = [p.energy for p in p_list]
        #print([p.t for p in p_list])
        #phi_list = [p.t*model.rf_program.get_frequency(p.t) for p in p_list]
        #print([phi-math.floor(phi) for phi in phi_list])
        #print([math.sin(2*math.pi*phi) for phi in phi_list])
        #print(t_list)
        #print(e_list)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1,  position=[0.3, 0.3, 0.6, 0.6])
        if not self.s:
            self.s = utilities.matplot_marker_size(t_list)
        axes.set_title("Turn "+suffix)
        axes.set_ylabel("E [MeV]")
        if self.t_bins:
            axes.set_xlim(self.t_bins[0], self.t_bins[-1])
        axes.set_ylim(self.e_range[0], self.e_range[-1])
        if plot_contours:
            self.plot_contours(p_list[0], model, axes)
        axes.scatter(t_list, e_list, s=self.s)
        self.plot_energy(axes, model, p_list)
        axes.scatter(t_list[0], e_list[0], color="red")
        axes = figure.add_subplot(1, 1, 1,  position=[0.3, 0.1, 0.6, 0.2])
        axes.set_xlabel("$[t_i*f_0 - floor(t_i*f_0)] \\times 360$")
        axes.hist(t_list, bins=self.t_bins)
        if self.t_bins:
            axes.set_xlim(self.t_bins[0], self.t_bins[-1])
        figure.savefig(os.path.join(self.output_directory, "longitudinal_"+suffix+".png"))
        return figure

    def t_norm(self, particle, ref_particle, model):
        #return particle.t
        dt = (particle.t-ref_particle.t)*model.rf_program.get_frequency(ref_particle.t)
        dt -= math.floor(dt)
        return dt*360

    def plot_energy(self, axes, model, p_list):
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()
        tolerance = (y_lim[1]-y_lim[0])*1e-3
        energy_ref = model.get_reference_energy(p_list[0].t, 1, BeamFactory.mass, p_list[0].energy, tolerance)
        print("Reference energy", model.get_time_of_flight(p_list[0]), 1/model.rf_program.get_frequency(p_list[0].t), energy_ref)
        axes.plot([x_lim[0], x_lim[1]], [energy_ref, energy_ref], c='orange')
        axes.set_xlim(x_lim)

    def plot_contours(self, ref_particle, model, axes):
        n_turns = 500
        ref_time = ref_particle.t
        ref_energy = ref_particle.energy
        frequency = model.rf_program.get_frequency(ref_particle.t)
        voltage = model.rf_program.get_voltage(ref_particle.t)
        t0 = 1./frequency
        test_model = copy.deepcopy(model)
        test_model.rf_program = ConstantBucket()
        test_model.rf_program.v0 = voltage
        test_model.rf_program.f0 = frequency
        for i in range(5):
            test_particle = copy.deepcopy(ref_particle)
            t = ref_time+(i+0.5)/4*t0
            test_particle.t = t
            t_list, e_list = [None]*n_turns, [None]*n_turns
            for i in range(n_turns):
                test_model.one_turn(test_particle)
                t_list[i] = self.t_norm(test_particle, ref_particle, test_model)
                e_list[i] = test_particle.energy
            axes.scatter(t_list, e_list, s=1)

class TurnAction(object):
    def __init__(self, rf_program, monitor, model, contours):
        self.program = rf_program
        self.monitor = monitor
        self.model = model
        self.plot_contours = contours
        self.plot_frequency = 100
        self.output_directory = "output/"

    def do_turn_action(self, turn, particle_collection):
        suffix = str(turn).rjust(4, "0")
        plotter = PlotBeam(self.output_directory)
        if turn % self.plot_frequency == 0:
            figure = plotter.plot_beam(particle_collection, self.model, self.plot_contours, suffix)
            matplotlib.pyplot.close(figure)
        self.monitor.do_one_turn(turn, particle_collection)
        print("Turn", turn)

def main_constant_bucket():
    utilities.clear_dir("output")
    model = LongitudinalModel()
    #p_list = BeamFactory.make_coasting_beam_square(10000, 20, 20.1, model)
    e0 = 19.5
    p0 = Particle(0, e0, BeamFactory.mass)
    t0 = model.get_time_of_flight(p0)
    p_list = [p0, Particle(0, e0-0.1, BeamFactory.mass), Particle(0, e0+0.1, BeamFactory.mass), ]
    p_list = [Particle(t0/2, e0, BeamFactory.mass), Particle(-t0/2, e0, BeamFactory.mass), ]+p_list
    program = ConstantBucket()
    program.f0 = 1.0/t0
    program.v0 = 0.00
    model.rf_program = program
    monitor = BeamMonitor()
    model.do_turn_action = TurnAction(program, monitor, model, True).do_turn_action
    model.track_beam(max_time = 2e5, max_turn = None, particle_collection=p_list)
    #monitor.do_plot(10)

def main_frequency_ramp():
    model = LongitudinalModel()
    p_start = Particle(0, 19.5, BeamFactory.mass)
    p_end = Particle(0, 24.0, BeamFactory.mass)
    t0 = model.get_time_of_flight(p_start)
    t1 = model.get_time_of_flight(p_end)
    ramp_time = t1*100

    p_mid = Particle(t0/2.0, (21+22)/2, BeamFactory.mass)
    p_list = [p_mid]+BeamFactory.make_coasting_beam_square(10000, 21, 22, model)
    #p_list = [p_mid]+[Particle(t0*i/10.0, (21+22)/2, BeamFactory.mass) for i in range(3)]
    #p_list = [p_mid]+BeamFactory.make_uniform_distribution(10000, 0, t0, 21, 22)
    program = PiecewiseInterpolation()
    program.f_list = [1.0/t0, 1.0/t0, 1.0/t1, 1.0/t1]
    #program.f_list = [1.0/t0, 1.0/t1]
    program.t_list = [ramp_time*i for i in range(len(program.f_list))]
    max_time = ramp_time*(len(program.f_list)-1)
    program.v_list = [0.004]*len(program.t_list)
    program.setup(max_time)
    model.rf_program = program
    monitor = BeamMonitor()
    output_directory = f"output/test_{len(program.f_list)}"
    utilities.clear_dir(output_directory)

    turn_action = TurnAction(program, monitor, model, False)
    turn_action.plot_frequency = 10
    turn_action.output_directory = output_directory
    model.do_turn_action = turn_action.do_turn_action
    model.track_beam(max_time = max_time, max_turn = None, particle_collection=p_list)
    figure = monitor.do_plot(10, output_directory, model)
    model.write_rf_data(output_directory+"/rf.dat", 10.0, max_time)

def test():
    model = LongitudinalModel()
    # 800 mm orbit excursion
    for energy in [11, 19.5, 20, 21, 24, 150]:
        part = Particle(0, energy, BeamFactory.mass)
        print(energy, model.path_length(part)/2.0/math.pi)


if __name__ == "__main__":
    #main_constant_bucket()
    main_frequency_ramp()
    #test()
    #matplotlib.pyplot.show(block=False)
    #input("Press <CR> to finish")


