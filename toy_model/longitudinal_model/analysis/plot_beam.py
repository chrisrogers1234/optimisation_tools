import os
import math

import numpy
import matplotlib
import matplotlib.pyplot

import optimisation_tools.utils.utilities as utilities
import optimisation_tools.toy_model.longitudinal_model.analysis as analysis
import optimisation_tools.toy_model.longitudinal_model.beam as beam
import beam.particle
import analysis.plot_contours

class PlotBeam(object):
    def __init__(self, output_directory, model):
        self.model = model
        self.t_bins = [model.harmonic_number*i*360.0/100 for i in range(0, 100+1)]
        self.e_range = [69.75, 70.25]
        self.s = None
        self.output_directory = output_directory

    def plot_beam(self, p_list, plot_contours, suffix):
        if len(p_list) == 0:
            return
        t_list = [self.t_norm(p.t, p_list[0]) for p in p_list]
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
            analysis.plot_contours.PlotContours(p_list[0], self.model, axes).plot_contours()
        axes.scatter(t_list, e_list, s=self.s)
        self.plot_energy(axes, self.model, p_list)
        axes.scatter(t_list[0], e_list[0], color="red")
        # time projection
        axes = figure.add_subplot(1, 1, 1,  position=[0.1, 0.1, 0.6, 0.2])
        axes.set_xlabel("$[t_i*f_0 - floor(t_i*f_0)] \\times 360$")
        t_hist, t_bin_list, patches = axes.hist(t_list, bins=self.t_bins)
        min_t, max_t = min(t_hist), max(t_hist)
        axes.text(0.8, 0.1, f"min, max: {min_t:.1f}, {max_t:.1f}",
                  transform=axes.transAxes)
        self.plot_echange(axes, p_list[0], self.model)
        axes.set_xlim(self.t_bins[0], self.t_bins[-1])

        axes = figure.add_subplot(1, 1, 1,  position=[0.7, 0.3, 0.3, 0.6])
        e_hist, e_bins = numpy.histogram(e_list, 1000, self.e_range)
        axes.barh(e_bins[:-1], e_hist, (e_bins[1]-e_bins[0]), align="center")
        min_e, max_e, mean_e = min(e_list[1:]), max(e_list[1:]), numpy.mean(e_list[1:])
        axes.text(0.1, 0.95, f"min, mean, max: {min_e:.5f}, {mean_e:.5f}, {max_e:.5f} [MeV]",
                  transform=axes.transAxes)
        figure.savefig(os.path.join(self.output_directory, "longitudinal_"+suffix+".png"))
        return figure

    def t_norm(self, time, ref_particle):
        #return particle.t
        dt = (time-ref_particle.t)*self.model.rf_program.get_frequency(ref_particle.t)+self.model.harmonic_number/2
        dt -= self.model.harmonic_number*math.floor(dt/self.model.harmonic_number)
        return dt*360

    def t_norm_inverse(self, dphi_deg, ref_particle):
        dt = dphi_deg/360.0-self.model.harmonic_number/2
        dt = dt/self.model.rf_program.get_frequency(ref_particle.t)
        dt += ref_particle.t
        return dt

    def plot_echange(self, axes, ref_particle, model):
        # t_bins is angle in deg relative to RF
        e0 = ref_particle.energy
        m0 = ref_particle.mass
        f0 = model.rf_program.get_frequency(ref_particle.t)
        actual_t_list = [self.t_norm_inverse(t, ref_particle) for t in self.t_bins]
        echange = [model.get_rf_energy_change(beam.particle.Particle(actual_t, e0, m0)) for actual_t in actual_t_list]
        min_e, max_e = min(echange), max(echange)
        min_a, max_a = axes.get_ylim()
        norm = lambda e: e #(max_a-min_a)/(max_e-min_e)*(e-min_e)+min_a
        echange = [norm(e) for e in echange]

        e0 = model.get_reference_energy(ref_particle.t, ref_particle.energy, 1e-12)
        e1 = model.get_reference_energy(ref_particle.t+1/f0, ref_particle.energy, 1e-12)
        de_axes = axes.twinx()
        de_axes.set_position(axes.get_position())
        de_axes.plot([self.t_bins[0], self.t_bins[-1]], [norm(0.0)]*2, color="lightgrey", linestyle="--")
        de_axes.plot([self.t_bins[0], self.t_bins[-1]], [norm(e1-e0)]*2, color="lightgrey", linestyle="dotted")

        de_axes.plot(self.t_bins, echange, color="orange")

        ref_t = [self.t_norm(ref_particle.t, ref_particle)]
        ref_echange = [norm(model.get_rf_energy_change(ref_particle))]
        de_axes.set_ylabel("dE per turn [MeV]")
        de_axes.scatter(ref_t, ref_echange, color="red")


    def plot_energy(self, axes, model, p_list):
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()
        tolerance = (y_lim[1]-y_lim[0])*1e-3
        energy_ref = model.get_reference_energy(p_list[0].t, p_list[0].energy, tolerance)
        print("Reference energy", model.get_time_of_flight(p_list[0]), 1/model.rf_program.get_frequency(p_list[0].t), energy_ref)
        axes.plot([x_lim[0], x_lim[1]], [energy_ref, energy_ref], c='orange')
        axes.set_xlim(x_lim)