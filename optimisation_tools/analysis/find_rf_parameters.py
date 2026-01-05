"""
Script to find the RF set up; drives find closed orbit
"""

import os
import sys
import copy
import json
import math

import numpy
import ROOT

import xboa.common
from xboa.hit import Hit
sys.path.insert(1, "scripts")
from optimisation_tools.opal_tracking import OpalTracking
from . import find_closed_orbits
from optimisation_tools.plotting import plot_dump_fields
from optimisation_tools.utils import utilities

class FindRFParameters(object):
    def __init__(self, config):
        self.config = config
        self.energy_time_list = []
        self.closed_orbit_list = []
        self.et_fit = []
        self.freq_fit = []
        self.phase_fit = []
        self.reference_et = []
        self.mass = xboa.common.pdg_pid_to_mass[self.config.tracking["pdg_pid"]]
        self.canvases = {}
        self.output_dir = self.config.run_control["output_dir"]
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.find_rf_parameters["run_dir"])
        self.e0 = self.config.find_rf_parameters["start_energy"]
        print(self.tmp_dir)

    def find_rf_parameters(self):
        print("Finding RF parameters")
        if self.config.find_rf_parameters["do_co_scan"]:
            self.find_closed_orbits()
        self.load_closed_orbits()
        print(self.energy_time_list)
        self.fit_energy_time_relation()
        self.find_ref_phase()
        self.summary()

    def find_closed_orbits(self):
        e1 = self.config.find_rf_parameters["end_energy"]
        n_steps = self.config.find_rf_parameters["n_steps"]
        de_list = self.config.find_rf_parameters["delta_energy_list"]
        if n_steps > 1:
            e_step = (e1-self.e0)/(n_steps-1)
        else:
            e_step = 1.
        energy_list = [e_step*i + self.e0 for i in range(n_steps)]+de_list
        energy_list = sorted(energy_list)
        for i, ref_subs in enumerate(self.config.substitution_list):
            co_config = copy.deepcopy(self.config)
            co_config.run_control["output_dir"] += "/find_rf_closed_orbits/"
            try:
                os.makedirs(co_config.run_control["output_dir"])
            except OSError:
                pass
            co_config.find_closed_orbits["subs_overrides"] = self.config.find_rf_parameters["phasing_subs_overrides"]
            co_config.find_closed_orbits["do_plot_orbit"] = False
            co_config.find_closed_orbits["run_dir"] = co_config.find_rf_parameters["run_dir"]
            co_config.find_closed_orbits["output_file"] = "find_rf_time_"+str(i)
            new_sub_list = []
            for energy in energy_list:
                subs = copy.deepcopy(ref_subs)
                subs["__energy__"] = energy
                new_sub_list.append(subs)
            co_config.substitution_list = new_sub_list
            find_closed_orbits.CONFIG = None
            find_closed_orbits.OUT_DIR = None
            find_closed_orbits.RUN_DIR = None
            find_closed_orbits.main(co_config)

    def load_closed_orbits(self):
        for i, ref_subs in enumerate(self.config.substitution_list):
            fin = open(self.config.run_control["output_dir"]+"/find_rf_closed_orbits/find_rf_time_"+str(i)+".out")
            co_list = [json.loads(line) for line in fin.readlines()]
            et_list = []
            self.closed_orbit_list.append([Hit.new_from_dict(item["hits"][0]) for item in co_list])
            for co in co_list:
                energy = co["hits"][0]["energy"] - self.mass
                hit_times = [hit["t"] for hit in co["hits"][1:]]
                delta_times = [t1-hit_times[i] for i, t1 in enumerate(hit_times[1:])]
                mean_time = numpy.mean(delta_times) #]
                delta_times = [time for time in delta_times if abs(time-mean_time) < mean_time/2.]
                print("One Cell Time", mean_time, "Sigma", numpy.std(delta_times))
                mean_time = numpy.mean(delta_times)*self.config.find_rf_parameters["n_cells"]
                et_list.append((energy, mean_time))
            self.energy_time_list.append(et_list)

    def mom_var(self, energy, index):
        k = self.config.substitution_list[index]["__field_index__"]
        energy += self.mass
        var = energy*(energy**2-self.mass**2)**(-k/2/(k+1))
        return var

    def fit_energy_time_relation(self):
        for i, et_list in enumerate(self.energy_time_list):
            et_list = tuple(et_list)
            energy, time = list(zip(*et_list))

            name = "chi vs time-of-flight "+str(i)
            var = [self.mom_var(E, i) for E in energy]
            # this should be linear for perfect scaling
            hist, graph = xboa.common.make_root_graph(name, var, "#chi=Ep^{#frac{-k}{k+1}}", list(time), "time-of-flight [ns]")
            canvas = xboa.common.make_root_canvas(name)
            hist.Draw()
            graph.SetMarkerStyle(20)
            graph.Draw("LP")
            self.et_fit.append(ROOT.TF1("fit", "pol1"))
            graph.Fit(self.et_fit[-1])
            name = name.replace(" ", "_")
            self.find_frequency_time_dependence(i)
            self.canvases["chi vs T "+str(i)] = canvas

            name = "f vs E "+str(i)
            freq = [1e3/t for t in time]
            hist, graph = xboa.common.make_root_graph(name, freq, "frequency [MHz]", list(energy), "E [MeV]")
            canvas = xboa.common.make_root_canvas(name)
            hist.Draw()
            graph.SetLineColor(ROOT.kBlue)
            graph.Draw("L")
            self.canvases["f vs e "+str(i)] = canvas


    def tof(self, energy, index):
        var = self.mom_var(energy, index)
        tof = self.et_fit[index].Eval(var)
        return tof

    def find_frequency_time_dependence(self, index):
        poly = self.config.find_rf_parameters["freq_polynomial"]
        time, energy = 0, self.config.find_rf_parameters["start_energy"]
        time_list, frequency_list, energy_list = [time], [1./self.tof(energy, index)], [energy]
        while energy < self.config.find_rf_parameters["end_energy"]:
            time += self.tof(energy, index)
            energy += self.config.find_rf_parameters["v_peak"]*1e-3*math.sin(self.config.find_rf_parameters["phi_s"])
            energy_list.append(energy)
            time_list.append(time)
            frequency_list.append(1.e3/self.tof(energy, index))
        self.reference_et.append([time_list, energy_list])
        name = "time vs frequency "+str(index)
        hist, graph = xboa.common.make_root_graph(name, time_list[:-1], "t [ns]", frequency_list[1:], "frequency [MHz]")
        canvas = xboa.common.make_root_canvas(name)
        hist.Draw()
        graph.SetMarkerStyle(20)
        graph.SetLineColor(ROOT.kBlue)
        graph.Draw("L")
        self.freq_fit.append(ROOT.TF1("fit", "pol"+str(poly)))
        self.freq_fit[-1].SetLineColor(ROOT.kGray)
        self.freq_fit[-1].FixParameter(0, frequency_list[1]);
        graph.Fit(self.freq_fit[-1])
        self.canvases["t vs f "+str(index)] = canvas

        name = "time vs energy "+str(index)
        hist, graph = xboa.common.make_root_graph(name, time_list, "t [ns]", energy_list, "energy [MeV]")
        canvas = xboa.common.make_root_canvas(name)
        hist.Draw()
        graph.SetLineColor(ROOT.kBlue)
        graph.Draw("L")
        self.canvases["t vs e "+str(index)] = canvas

    def setup_subs_for_rf(self, index, phase, polynomial_coefficients):
        if polynomial_coefficients == None:
            poly = self.config.find_rf_parameters["freq_polynomial"]
            polynomial_coefficients = [self.freq_fit[index].GetParameter(i) for i in range(poly+1)]
        if phase == None:
            phase = -self.phase_fit[index].GetParameter(0)+self.config.find_rf_parameters["phi_s"]
        while len(polynomial_coefficients) < 4:
            polynomial_coefficients.append(0.)
        subs = {
            "__rf_voltage__":self.config.find_rf_parameters["v_peak"],
            "__rf_phase__":phase,
            "__rf_freq_0__":polynomial_coefficients[0], # MHz
            "__rf_freq_1__":polynomial_coefficients[1]/2., # MHz/ns
            "__rf_freq_2__":polynomial_coefficients[2]/3., # MHz/ns^2
            "__rf_freq_3__":polynomial_coefficients[3], # MHz/ns^3
        }
        return subs

    def _temp_dir(self):
        try:
            os.makedirs(self.tmp_dir)
        except OSError:
            pass
        os.chdir(self.tmp_dir)


    def track_one(self, index, subs_overrides_list, kinetic_energy = None):
        if kinetic_energy == None:
            kinetic_energy = self.e0 
        subs = self.config.substitution_list[index]
        for sub_dict in subs_overrides_list:
            for item, key in sub_dict.items():
                subs[item] = key

        print("Tracking with", end=' ')
        for key in sorted(subs.keys()):
            print(utilities.sub_to_name(key), subs[key], end=' ')
        print()
        self._temp_dir() # changes to tmp/find_rf_parameters/
        xboa.common.substitute(
            self.config.tracking["lattice_file"],
            "SectorFFAGMagnet.tmp",
            subs)
        ref_hit = sorted(self.closed_orbit_list[index],
                          key=lambda hit: abs(hit["kinetic_energy"] - self.e0) 
                        )[0]
        test_hit = sorted(self.closed_orbit_list[index],
                          key=lambda hit: abs(hit["kinetic_energy"] - kinetic_energy) 
                         )[0]
        ref_hit =  ref_hit.deepcopy()
        test_hit = test_hit.deepcopy()
        momentum = ref_hit["p"]
        ref_hit["x"] = 0.
        ref_hit["px"] = 0.
        ref_hit["p"] = momentum
        ref_hit.mass_shell_condition("p")
        tracking = OpalTracking("SectorFFAGMagnet.tmp",
                                "disttest.dat",
                                ref_hit,
                                self.config.find_rf_parameters["probe_files"],
                                self.config.tracking["opal_path"],
                                "log")
        print("Tracking hit with kinetic energy:", test_hit["kinetic_energy"])
        hit_list = tracking.track_one(test_hit)
        return hit_list

    def find_ref_phase(self):
        for index, fit in enumerate(self.freq_fit):
            phasing_list = []
            n_steps = 11
            for phase_index in range(n_steps):
                phase = 2.*math.pi*phase_index/(n_steps-1)
                e0 = self.config.find_rf_parameters["start_energy"]
                subs_list = [
                    self.config.find_rf_parameters["phasing_subs_overrides"],
                    self.setup_subs_for_rf(index, phase, None),
                ]
                print(subs_list)
                hit = self.track_one(index, subs_list)[2]
                phasing_list.append((phase, hit["energy"] - hit["mass"]))
            phase, energy = list(zip(*tuple(phasing_list)))
            name = "rf signal "+str(index)
            hist, graph = xboa.common.make_root_graph(name, list(phase), "#phi [rad]", list(energy), "E [MeV]")
            canvas = xboa.common.make_root_canvas(name)
            canvas.Draw()
            hist.Draw()
            graph.SetMarkerStyle(20)
            graph.Draw("LP")
            v_peak = self.config.find_rf_parameters["v_peak"]*1e-3
            self.phase_fit.append(ROOT.TF1(name, str(v_peak)+"*sin(x+[0])+"+str(e0)+"+[1]"))
            graph.Fit(self.phase_fit[-1])
            canvas.Update()
            name = name.replace(" ", "_")
            canvas.Print(name+".png")

    def summary(self):
        for index, fit in enumerate(self.freq_fit):
            self.canvases["x vs e "+str(index)] = xboa.common.make_root_canvas("x vs e "+str(index))
            self.canvases["dt vs de "+str(index)] = xboa.common.make_root_canvas("dt vs de "+str(index))
            first = True
            for ke in [self.e0]+self.config.find_rf_parameters["delta_energy_list"]:
                subs_list = [
                    self.config.find_rf_parameters["final_subs_overrides"],
                    self.setup_subs_for_rf(index, None, None) # offset time on the tracked particle, not the RF cavity
                ]
                hit_list = self.track_one(index, subs_list, ke)
                et_tuple = ((hit["t"], hit["kinetic_energy"]) for hit in hit_list)
                my_var = ["station", "t", "x", "px", "y", "z", "pz", "kinetic_energy"] 
                for var in my_var:
                    print(var.rjust(8), end=' ')
                print()
                i = 0
                while i < 10 and i < len(hit_list):
                    for var in my_var:
                        print(str(round(hit_list[i][var], 3)).rjust(8), end=' ')
                    print()
                    i += 1
                time, energy = list(zip(*et_tuple))
                name = "reference acceleration t vs E "+str(index)
                hist, graph = xboa.common.make_root_graph(name, list(time), "t [ns]", list(energy), "E [MeV]")
                canvas = self.canvases["t vs e "+str(index)]
                canvas.cd()
                if not first:
                    graph.SetLineColor(ROOT.kGreen+2)
                graph.Draw("L")

                name = "reference acceleration t vs E "+str(index)
                hist, graph = xboa.common.make_root_graph(name, list(time), "t [ns]", list(energy), "E [MeV]")
                canvas = self.canvases["t vs e "+str(index)]
                canvas.cd()
                if not first:
                    graph.SetLineColor(ROOT.kGreen+2)
                graph.Draw("L")

                name = "reference acceleration dt vs dE "+str(index)
                if first:
                    ref_time, ref_energy = time, energy
                dt_list = [time[i] - t for i, t in enumerate(ref_time) if i < len(time) and i < len(ref_time)]
                de_list = [energy[i] - E for i, E in enumerate(ref_energy) if i < len(energy) and i < len(ref_energy)]
                hist, graph = xboa.common.make_root_graph(name, dt_list, "t [ns]", de_list, "E [MeV]",
                                                          xmin=-2000.0, xmax=+2000.0, ymin=-2.0, ymax=+2.0)
                canvas = self.canvases["dt vs de "+str(index)]
                canvas.cd()
                if first:
                    hist.Draw()
                else:
                    graph.SetMarkerColor(ROOT.kGreen+2)

                graph.SetMarkerStyle(7)
                graph.Draw("P")

                delta_time = [1.e3/(t - time[i+1]) for i, t in enumerate(time[2:])]
                name = "reference acceleration t vs f "+str(index)
                hist, graph = xboa.common.make_root_graph(name, list(time)[2:], "t [ns]", delta_time, "f [MHz]")
                canvas = self.canvases["t vs f "+str(index)]
                canvas.cd()
                if not first:
                    graph.SetLineColor(ROOT.kGreen+2)
                graph.Draw("L")

                name = "reference acceleration f vs E "+str(index)
                hist, graph = xboa.common.make_root_graph(name, delta_time, "freq [MHz]", list(energy)[2:], "E [MeV]")
                canvas = self.canvases["f vs e "+str(index)]
                canvas.cd()
                if not first:
                    graph.SetLineColor(ROOT.kGreen+2)
                graph.Draw("L")

                ex_tuple = ((hit["x"], hit["kinetic_energy"]) for hit in hit_list)
                time, energy = list(zip(*ex_tuple))
                name = "reference acceleration x vs E "+str(index)
                hist, graph = xboa.common.make_root_graph(name, list(time), "x [mm]", list(energy), "E [MeV]")
                canvas = self.canvases["x vs e "+str(index)]
                canvas.cd()
                if first:
                    hist.Draw()
                else:
                    graph.SetLineColor(ROOT.kGreen+2)
                graph.Draw("L")

                first = False
            try:
                rf_parameters = plot_dump_fields.main(self.tmp_dir)
            except Exception:
                sys.excepthook(*sys.exc_info())
                rf_parameters = []
            self.canvases["t vs f "+str(index)].cd()
            time_list = [data["t0"] for data in rf_parameters]
            freq_list = [data["frequency"]*1e3 for data in rf_parameters]
            hist, graph = xboa.common.make_root_graph("f vs t fit", time_list, "t [ns]", freq_list, "f [MHz]")
            graph.SetMarkerStyle(20)
            graph.SetMarkerColor(ROOT.kRed)
            graph.Draw("P")
            canvas.Update()
            for key, canvas in self.canvases.items():
                canvas.cd()
                canvas.Update()
                for fmt in "root", "png", "pdf":
                    name = self.output_dir+"/rf_"+key.replace(" ", "_")
                    canvas.Print(name+"."+fmt)


def main(config):
    find_rf = FindRFParameters(config)
    find_rf.find_rf_parameters()
    input()
