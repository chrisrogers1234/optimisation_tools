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

class LongitudinalModel(object):
    def __init__(self):
        self.r0 = 4540 # mm
        self.k = 7.45
        self.phi0 = 0.0 # rad
        self.mass = BeamFactory.mass # MeV/c^2
        self.c_light = 300 # mm/ns
        self.injection_momentum = 11 # MeV/c
        self.rf_program = ConstantBucket()

    def path_length(self, a_particle):
        """Return path length in mm"""
        p0 = ((self.injection_momentum+BeamFactory.mass)**2-BeamFactory.mass**2)**0.5 # injection momentum
        p1 = a_particle.momentum()
        r1 = self.r0*(p1/p0)**(1/(self.k+1)) # k+1 from b r = p q; note radial dependence
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
            #print(energy_estimate, new_energy_estimate)
            energy_estimate = new_energy_estimate
        return energy_estimate

    def one_turn(self, a_particle):
        dt = self.get_time_of_flight(a_particle)
        a_particle.t += dt
        dE = self.get_rf_energy_change(a_particle)
        a_particle.energy += dE
        return a_particle

    def get_ref_delta_energy(self, test_particle, harmonic_number):
        f0 = self.rf_program.get_frequency(test_particle.t)
        e0 = self.get_reference_energy(test_particle.t, harmonic_number, test_particle.mass, test_particle.energy, 1e-12)
        e1 = self.get_reference_energy(test_particle.t+1/f0, harmonic_number, test_particle.mass, test_particle.energy, 1e-12)
        return e1-e0

    def set_to_reference_particle(self, a_particle):
        """Move the particle to the reference point, in the same rf period"""
        dE = self.get_ref_delta_energy(a_particle, 1)
        dt = self.rf_program.get_relative_time(a_particle.t) # relative to 0 crossing
        f0 = self.rf_program.get_frequency(a_particle.t)
        v0 = self.rf_program.get_voltage_magnitude(a_particle.t)
        reference_phase = math.asin(dE/v0)
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

    def get_voltage_magnitude(self, t):
        return self.v_interpolation(t)

    def get_relative_time(self, t):
        index = bisect.bisect_left(self.lookup_time, t)
        try:
            dt = t-self.lookup_time[index]
        except IndexError:
            dt = t-self.lookup_time[-1]
        return dt

    def get_voltage(self, t):
        dt = self.get_relative_time(t)
        freq = self.f_interpolation(t)
        v0 = self.v_interpolation(t)
        volts = v0*math.sin(2*math.pi*freq*dt)
        return volts

    def get_frequency(self, t):
        return self.f_interpolation(t)

class Particle(object):
    def __init__(self, t, energy, mass=None):
        self.t = t
        self.energy = energy
        self.mass = mass
        if self.mass == None:
            self.mass = BeamFactory.mass

    def momentum(self):
        total_energy = self.energy + self.mass
        momentum = (total_energy**2-self.mass**2)**0.5
        return momentum

    def beta(self):
        if self.energy <= 0.0:
            return 0.0
        total_energy = self.energy + self.mass
        momentum = (total_energy**2-self.mass**2)**0.5
        beta = momentum/total_energy
        return beta

class BeamFactory(object):
    def __init__(self):
        pass

    @classmethod
    def make_coasting_beam_square(cls, n_events, low_energy, high_energy, n_turns, model):
        tof_list = []
        e_list = numpy.random.uniform(low_energy, high_energy, n_events)
        p_list = [Particle(0, e, cls.mass) for e in e_list]
        for p in p_list:
            tof = model.get_time_of_flight(p)
            p.t = numpy.random.uniform(0.0, tof*n_turns)
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
        self.e_range = [56.0, 59.0]
        self.s = None
        self.output_directory = output_directory

    def plot_beam(self, p_list, model, plot_contours, suffix):
        if len(p_list) == 0:
            return
        t_list = [self.t_norm(p.t, p_list[0], model) for p in p_list]
        e_list = [p.energy for p in p_list]
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        axes = figure.add_subplot(1, 1, 1,  position=[0.1, 0.3, 0.6, 0.6])
        if not self.s:
            self.s = utilities.matplot_marker_size(t_list)
        axes.set_title("Turn "+suffix)
        axes.set_ylabel("E [MeV]")
        axes.set_xlim(self.t_bins[0], self.t_bins[-1])
        axes.set_ylim(self.e_range[0], self.e_range[-1])
        if plot_contours:
            PlotContours(p_list[0], model, axes).plot_contours()
        axes.scatter(t_list, e_list, s=self.s)
        self.plot_energy(axes, model, p_list)
        axes.scatter(t_list[0], e_list[0], color="red")
        # time projection
        axes = figure.add_subplot(1, 1, 1,  position=[0.1, 0.1, 0.6, 0.2])
        axes.set_xlabel("$[t_i*f_0 - floor(t_i*f_0)] \\times 360$")
        axes.hist(t_list, bins=self.t_bins)
        self.plot_echange(axes, p_list[0], model)
        axes.set_xlim(self.t_bins[0], self.t_bins[-1])

        axes = figure.add_subplot(1, 1, 1,  position=[0.7, 0.3, 0.3, 0.6])
        e_hist, e_bins = numpy.histogram(e_list, 1000, self.e_range)
        axes.barh(e_bins[:-1], e_hist, (e_bins[1]-e_bins[0])/2, align="edge")
        figure.savefig(os.path.join(self.output_directory, "longitudinal_"+suffix+".png"))
        return figure

    def t_norm(self, time, ref_particle, model):
        #return particle.t
        dt = (time-ref_particle.t)*model.rf_program.get_frequency(ref_particle.t)+0.5
        dt -= math.floor(dt)
        return dt*360

    def t_norm_inverse(self, dphi_deg, ref_particle, model):
        dt = dphi_deg/360.0-0.5
        dt = dt/model.rf_program.get_frequency(ref_particle.t)
        dt += ref_particle.t
        return dt

    def plot_echange(self, axes, ref_particle, model):
        # t_bins is angle in deg relative to RF
        e0 = ref_particle.energy
        m0 = ref_particle.mass
        f0 = model.rf_program.get_frequency(ref_particle.t)
        actual_t_list = [self.t_norm_inverse(t, ref_particle, model) for t in self.t_bins]
        echange = [model.get_rf_energy_change(Particle(actual_t, e0, m0)) for actual_t in actual_t_list]
        min_e, max_e = min(echange), max(echange)
        min_a, max_a = axes.get_ylim()
        norm = lambda e: (max_a-min_a)/(max_e-min_e)*(e-min_e)+min_a
        echange = [norm(e) for e in echange]

        e0 = model.get_reference_energy(ref_particle.t, 1, BeamFactory.mass, ref_particle.energy, 1e-12)
        e1 = model.get_reference_energy(ref_particle.t+1/f0, 1, BeamFactory.mass, ref_particle.energy, 1e-12)

        axes.plot([self.t_bins[0], self.t_bins[-1]], [norm(0.0)]*2, color="lightgrey", linestyle="--")
        axes.plot([self.t_bins[0], self.t_bins[-1]], [norm(e1-e0)]*2, color="lightgrey", linestyle="dotted")

        axes.plot(self.t_bins, echange)

        ref_t = [self.t_norm(ref_particle.t, ref_particle, model)]
        ref_echange = [norm(model.get_rf_energy_change(ref_particle))]
        axes.scatter(ref_t, ref_echange, color="red")

    def plot_energy(self, axes, model, p_list):
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()
        tolerance = (y_lim[1]-y_lim[0])*1e-3
        energy_ref = model.get_reference_energy(p_list[0].t, 1, BeamFactory.mass, p_list[0].energy, tolerance)
        print("Reference energy", model.get_time_of_flight(p_list[0]), 1/model.rf_program.get_frequency(p_list[0].t), energy_ref)
        axes.plot([x_lim[0], x_lim[1]], [energy_ref, energy_ref], c='orange')
        axes.set_xlim(x_lim)

class PlotContours(object):
    def __init__(self, ref_particle, model, axes):
        """ref particle *must* be on the stable fixed point"""
        self.model = model
        self.t0 = ref_particle.t
        self.f0 = model.rf_program.get_frequency(self.t0)
        self.v0 = model.rf_program.get_voltage_magnitude(self.t0)
        self.ref_energy = model.get_reference_energy(self.t0, 1, BeamFactory.mass, ref_particle.energy, 1e-12)
        self.axes = axes
        self.get_phis()
        print("Set up contour plot with t", self.t0, "f0", self.f0, "v0", self.v0, "v0 phis", self.v0_sin_phi_s)


    def get_phis(self):
        ref_energy_2 = self.model.get_reference_energy(self.t0+1/self.f0, 1, BeamFactory.mass, self.ref_energy, 1e-12)
        ref_energy_0 = self.model.get_reference_energy(self.t0-1/self.f0, 1, BeamFactory.mass, self.ref_energy, 1e-12)
        delta_energy = (ref_energy_2-ref_energy_0)/2 # change in energy in one RF period s.t. TOF = RF period
        if delta_energy > self.v0:
            print("ERROR: 2 turn reference energy sweep from", ref_energy_0, "to", ref_energy_2, "more than the RF voltage", self.v0, "so no RF exists. Generating on-crest.")
            self.v0_sin_phi_s = v0
        else:
            self.v0_sin_phi_s = delta_energy
        self.phi_s = math.asin(delta_energy/self.v0)

    def plot_contours(self):
        nx, ny = 6, 6
        xlim = self.axes.get_xlim()
        xlim = [math.radians(xlim[0]), math.radians(xlim[1])]
        ylim = self.axes.get_ylim()
        for yi in range(1, ny+2):
            ydelta = yi*(ylim[1]-ylim[0])/(ny+2)
            y0 = [xlim[0]+1e-9, ydelta+ylim[0]]
            if y0[1] < self.ref_energy:
                self.one_contour(y0, 1)
            else:
                self.one_contour(y0, -1)
        for xi in range(1, nx+2):
            phi0 = math.pi*2*xi/(nx+2)
            y0 = [phi0, self.ref_energy+1e-9]
            yi = self.one_contour(y0, -1)
            y0 = [phi0, self.ref_energy-1e-9]
            yi = self.one_contour(y0, 1)

    def one_contour(self, y0, direction):
        ebounds = self.axes.get_ylim()
        tbounds = self.axes.get_ylim()
        lower_ebound_event = lambda t, y, direction: y[1] - ebounds[0]
        upper_ebound_event = lambda t, y, direction: y[1] - ebounds[1]
        lower_tbound_event = lambda t, y, direction: y[0] - math.radians(0.0)
        upper_tbound_event = lambda t, y, direction: y[0] - math.radians(360.0)
        axis_crossing_event = lambda t, y, direction: y[1]-self.ref_energy
        events = [upper_tbound_event, lower_tbound_event, axis_crossing_event] #upper_ebound_event, lower_ebound_event, 
        for event in events:
            event.terminal = True
        if direction > 0:
            contour = scipy.integrate.solve_ivp(self.derivatives, (0, 1e3), y0, args=(direction,), events=events, first_step=0.01, max_step=5.0)
        else:
            contour = scipy.integrate.solve_ivp(self.derivatives, (0, -1e3), y0, args=(direction,), events=events, first_step=0.01, max_step=5.0)
        sorted_values = numpy.sort(contour.y, axis=0) # sort so that we don't get domain errors
        sorted_values = numpy.dot(numpy.array([[360/math.pi/2.0, 0.0], [0.0, 1.0]]), sorted_values) # convert to deg
        self.axes.plot(sorted_values[0], sorted_values[1], c="lightgrey")#, s=0.05)
        return sorted_values

    def derivatives(self, turn, yi, direction):
        ti = self.model.get_time_of_flight(Particle(0.0, yi[1]))
        dphidturn = 2*math.pi*(ti*self.f0-1)
        dEdturn = self.v0*math.sin(yi[0]+math.pi+self.phi_s) - self.v0_sin_phi_s # units are MeV
        dydx = (dphidturn, dEdturn)
        #print("Derivative at turn", turn, "yi", yi, "dti", (ti*self.f0-1), "dydturn", dydx)
        return dydx


class TurnAction(object):
    def __init__(self, rf_program, monitor, model, plot_contours):
        self.program = rf_program
        self.monitor = monitor
        self.model = model
        self.plot_contours = plot_contours
        self.plot_frequency = 100
        self.output_directory = "output/"
        self.plotter = PlotBeam(self.output_directory)

    def do_turn_action(self, turn, particle_collection):
        suffix = str(turn).rjust(4, "0")
        self.plotter.output_directory = self.output_directory
        if turn % self.plot_frequency == 0:
            figure = self.plotter.plot_beam(particle_collection, self.model, self.plot_contours, suffix)
            matplotlib.pyplot.close(figure)
        self.monitor.do_one_turn(turn, particle_collection)
        print("Turn", turn)

def main_frequency_ramp():
    model = LongitudinalModel()
    p_start = Particle(0, 56.7, BeamFactory.mass)
    p_end = Particle(0, 56.7, BeamFactory.mass)
    t0 = model.get_time_of_flight(p_start)
    t1 = model.get_time_of_flight(p_end)
    ramp_time = 0.1
    hold_time = 10

    p_mid = Particle(t0, p_start.energy, BeamFactory.mass)
    p_list = [p_mid]+BeamFactory.make_coasting_beam_square(10000, 57.7, 57.800001, n_turns=2, model=model)
    program = PiecewiseInterpolation()
    program.f_list = [1.0/t0, 1.0/t0, 1.0/t1, 1.0/t1]
    program.t_list = [0, t1*hold_time, t1*(hold_time+ramp_time), t1*(hold_time+ramp_time+hold_time)]
    max_time = program.t_list[-1]
    program.v_list = [0.004]*len(program.t_list)
    program.setup(max_time)
    model.rf_program = program
    monitor = BeamMonitor()
    output_directory = f"output/kurns_v4/hold_ramp_hold_{hold_time}_{ramp_time}_{hold_time}"
    utilities.clear_dir(output_directory)

    turn_action = TurnAction(program, monitor, model, plot_contours=True)
    turn_action.plot_frequency = 10
    turn_action.output_directory = output_directory
    model.do_turn_action = turn_action.do_turn_action
    model.track_beam(max_time = max_time, max_turn = None, particle_collection=p_list)
    print("Done tracking - finishing up")
    figure = monitor.do_plot(10, output_directory, model)
    model.write_rf_data(output_directory+"/rf.dat", 10.0, max_time)

def main_lemc():
    BeamFactory.mass = 105.658
    model = LongitudinalModel()
    model.path_length = lambda x: 100

    p_start = Particle(0, 1, BeamFactory.mass)
    p_mid = Particle(0, 2, BeamFactory.mass)
    p_end = Particle(0, 3, BeamFactory.mass)

    t0 = model.get_time_of_flight(p_start)
    t1 = model.get_time_of_flight(p_end)

    p_list = [p_mid]+BeamFactory.make_coasting_beam_square(100, p_start.energy, p_end.energy, n_turns=1, model=model)

    program = PiecewiseInterpolation()
    program.f_list = [0.333, 0.333]
    program.t_list = [0, 1e9]
    program.v_list = [1e-3, 1e-3] # MV
    program.setup(100.0)
    model.rf_program = program
    monitor = BeamMonitor()
    output_directory = f"output/cooling_v1/wandering_1"
    utilities.clear_dir(output_directory)

    turn_action = TurnAction(program, monitor, model, plot_contours=True)
    turn_action.plot_frequency = 1
    turn_action.output_directory = output_directory
    turn_action.plotter.e_range = [1, 10]
    model.do_turn_action = turn_action.do_turn_action
    model.track_beam(max_time = 30, max_turn = None, particle_collection=p_list)

if __name__ == "__main__":
    main_lemc()
    #test()
    #matplotlib.pyplot.show(block=False)
    #input("Press <CR> to finish")


