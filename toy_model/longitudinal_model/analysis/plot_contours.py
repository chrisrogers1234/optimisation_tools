import math

import numpy
import scipy
import matplotlib
import matplotlib.pyplot

import optimisation_tools.toy_model.longitudinal_model.beam as beam
import beam.particle

class PlotContours(object):
    def __init__(self, ref_particle, model, axes):
        """ref particle *must* be on the stable fixed point"""
        self.model = model
        self.t0 = ref_particle.t
        self.h0 = model.harmonic_number
        self.f0 = model.rf_program.get_frequency(self.t0)
        self.v0 = model.rf_program.get_voltage_magnitude(self.t0)
        self.ref_energy = model.get_reference_energy(self.t0, ref_particle.energy, 1e-12)
        print("REF ENERGY", self.ref_energy)
        self.axes = axes
        self.get_phis()
        print("Set up contour plot with t", self.t0, "f0", self.f0, "v0", self.v0, "v0 phis", self.v0_sin_phi_s)


    def get_phis(self):
        ref_energy_2 = self.model.get_reference_energy(self.t0+self.h0/self.f0, self.ref_energy, 1e-12)
        ref_energy_0 = self.model.get_reference_energy(self.t0-self.h0/self.f0, self.ref_energy, 1e-12)
        delta_energy = (ref_energy_2-ref_energy_0)/2 # change in energy in one RF period s.t. TOF = RF period
        if delta_energy > self.v0 and abs(self.v0) > 1e-9:
            print("ERROR: 2 turn reference energy sweep from", ref_energy_0, "to", ref_energy_2, "more than the RF voltage", self.v0, "so no RF exists. Generating on-crest.")
            self.v0_sin_phi_s = self.v0
        else:
            self.v0_sin_phi_s = delta_energy
        if abs(self.v0) > 1e-9:
            self.phi_s = math.asin(delta_energy/self.v0)
        else:
            self.phi_s = 0.0

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
        """Plot contour for particle starting at y0 = (phase, energy)"""
        ebounds = self.axes.get_ylim()
        tbounds = self.axes.get_ylim()
        lower_ebound_event = lambda t, y, direction: y[1] - ebounds[0]
        upper_ebound_event = lambda t, y, direction: y[1] - ebounds[1]
        lower_tbound_event = lambda t, y, direction: y[0] - math.radians(0.0)
        upper_tbound_event = lambda t, y, direction: y[0] - math.radians(360.0*self.h0)
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
        ti = self.model.get_time_of_flight(beam.particle.Particle(0.0, yi[1]))
        dphidturn = 2*math.pi*(ti*self.f0-self.h0)
        dEdturn = self.v0*math.sin(yi[0]+math.pi*self.h0+self.phi_s) - self.v0_sin_phi_s # units are MeV
        dydx = (dphidturn, dEdturn)
        return dydx
