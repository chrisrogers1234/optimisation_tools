import math
import bisect
import sys
import json

import numpy
from scipy.interpolate import interp1d
import ROOT
import xboa.common
from xboa.hit import Hit

class ToyLongitudinal(object):
    def __init__(self):
        self.scallop_factor = 1.0
        self.lookup_tof = None
        self.lookup_radius = None
        self.r0 = 250.
        self.p0 = 5.
        self.p1 = 150.
        self.n_turns = 100
        self.k = 1.0 #0.25
        self.n_stations = 1
        self.v_eff = 1e-2
        self.phi_s = 0.2
        self.omega = 2*math.pi*0.1/self.tof(self.p0)
        self.mass = xboa.common.pdg_pid_to_mass[2212]
        self.cavity_phase = [0., math.pi*0.5] # phase of cavity
        self.azimuthal_angle = [0.]+[0., 0.]+[1.] # position of cavity
        self.phase_list = [[], []] # phi_eff vs radius

        self.plot_dir = "./"
        self.output_dir = "./"

    def setup_lookup(self, filename, n_cells):
        json_in = [json.loads(line) for line in open(filename).readlines()]
        my_list = []
        for json_line in json_in:
            hits = [Hit.new_from_dict(item) for item in json_line['hits']]
            r = numpy.mean([hit['r'] for hit in hits])
            p = numpy.mean([hit['p'] for hit in hits])
            tof_list = [hits[i+1]['t']-hits[i]['t'] for i, hit in enumerate(hits[1:])]
            mean_tof = numpy.mean(tof_list)
            tof_list = [tof for tof in tof_list if abs(tof-mean_tof) < mean_tof/2.]
            mean_tof = numpy.mean(tof_list)
            print("Mean tof", mean_tof, "rms tof", numpy.std(tof_list))
            t = mean_tof*n_cells
            my_list.append((r, p, t))
        my_list = sorted(my_list)
        r_list, p_list, t_list = list(zip(*my_list))
        for i, p in enumerate(p_list):
            print(i, format(p, "8.4g"), format(r_list[i], "8.4g"), format(t_list[i], "8.4g"))
        self.lookup_tof = interp1d(p_list, t_list, kind='cubic')
        self.lookup_tof.bounds_error = False
        self.lookup_tof.fill_value = 'extrapolate'
        self.lookup_radius = interp1d(p_list, r_list, kind='cubic')
        self.lookup_radius.bounds_error = False
        self.lookup_radius.fill_value = 'extrapolate'

    def tof(self, momentum):
        if self.lookup_tof != None:
            return self.lookup_tof(momentum)
        radius = self.radius(momentum)
        energy = self.ke(momentum)+self.mass
        speed = momentum/energy*self.c_light
        circumference = math.pi*2.*radius*self.scallop_factor
        tof = circumference/speed
        return tof

    def ke(self, momentum):
        energy = (momentum**2+self.mass**2)**0.5
        ke = energy-self.mass
        return ke

    def radius(self, momentum):
        if self.lookup_radius != None:
            return self.lookup_radius(momentum)
        return self.r0*(momentum/self.p0)**(1./(self.k+1))

    def mean_bfield(self, momentum):
        bfield = momentum/self.c_light/self.radius(momentum)
        bfield*=1e3 # kT -> T
        return bfield

    def print_radius(self):
        p = self.p0
        title = ["p", "ke", "r", "By", "TOF"]
        for key in title:
            print(key.rjust(8), end=' ')
        print()
        while p <= self.p1+1e-6:
            value_list = [p, self.ke(p), self.radius(p), self.mean_bfield(p), self.tof(p)]
            for value in value_list:
                print(str(round(value, 2)).rjust(8), end=' ')
            print()
            p += (self.p1-self.p0)/27.

    def setup_phi_eff(self):
        """
        Calculate phi_n = sum(omega T_n) over n turns, applying time kick on each turn
        """
        momentum = self.p0
        time = 0.
        phi_eff = 0.
        for turn in range(self.n_turns+5):
            # evolve through one full revolution
            time += self.tof(momentum)
            self.phase_list[0].append(momentum)
            # phi_eff is the phase that a particle on the synchronous phase 
            # passes through the reference surface
            phi_eff = time*self.omega + 2.*math.pi*self.phi_s
            self.phase_list[1].append(phi_eff)
            # increment the energy
            energy = (momentum**2+self.mass**2)**0.5
            delta_energy = self.v_eff*math.sin(2.*math.pi*self.phi_s)
            energy += delta_energy
            momentum = (energy**2-self.mass**2)**0.5

    def get_phi_eff(self, momentum):
        """
        Returns the effective phase by linear interpolation
        """
        index = bisect.bisect_left(self.phase_list[0], momentum)
        if index >= len(self.phase_list[1]):
            index = len(self.phase_list[1])-1
        elif index == 0:
            index += 1
        # dp * Delta phi/Delta p + phi0
        dp = (momentum-self.phase_list[0][index-1])
        phi0 = self.phase_list[1][index-1]
        delta_p = self.phase_list[0][index]-self.phase_list[0][index-1]
        delta_phi = self.phase_list[1][index]-self.phase_list[1][index-1]
        phi_eff = dp*delta_phi/delta_p+phi0 # dp* dphi/dp + phi0
        return phi_eff

    def get_voltage(self, momentum, station):
        phi_eff = self.get_phi_eff(momentum)
        v = self.v_eff*math.sin(2.*math.pi*self.phi_s)
        dv_dt_omega = self.v_eff*math.cos(2.*math.pi*self.phi_s) # 1/omega dv/dt

        arg_1 = 2.*math.pi*self.cavity_phase[0]+phi_eff+self.omega * self.azimuthal_angle[1] * self.tof(momentum)
        arg_2 = 2.*math.pi*self.cavity_phase[1]+phi_eff+self.omega * self.azimuthal_angle[2] * self.tof(momentum)
        s_1 = math.sin(arg_1)
        s_2 = math.sin(arg_2)
        c_1 = math.cos(arg_1)
        c_2 = math.cos(arg_2)

        denominator = c_2*s_1 - s_2*c_1
        v_0 = (v*c_2 - dv_dt_omega*s_2)/denominator
        v_1 = (dv_dt_omega*s_1 - v*c_1)/denominator
        if station == 0:
            return max(v_0, 0)
        elif station == 1:
            return max(v_1, 0)
        elif station == 2:
            return max(-v_0, 0)
        elif station == 3:
            return max(-v_1, 0)
        else:
            raise ValueError("Station must be 0 or 1")

    def get_rf(self, momentum, time, station):
        omega = self.omega # fixed
        phase = self.cavity_phase[station] # fixed at each station
        voltage = self.get_voltage(momentum, station) # allowed to vary with momentum i.e. radius
        return omega, voltage, phase

    def energy_kick(self, momentum, time, rf_station):
        energy = (momentum**2+self.mass**2)**0.5
        omega, voltage, phase = self.get_rf(momentum, time, rf_station)
        delta = time*omega+2.*math.pi*phase
        energy += voltage*math.sin(delta)
        if energy < self.mass:
            momentum = 1e-6
        else:
            momentum = (energy**2-self.mass**2)**0.5
        return momentum

    def delta_energy(self, momentum, time, rf_station):
        omega, voltage, phase = self.get_rf(momentum, time, rf_station)
        delta = time*omega+2.*math.pi*phase
        delta_energy = voltage*math.sin(delta)
        return delta_energy

    def time_kick(self, momentum, time, rf_station):
        dtheta = self.azimuthal_angle[rf_station+1]-self.azimuthal_angle[rf_station]
        delta_time = self.tof(momentum)*dtheta
        time += delta_time
        return time

    def time_to_phase(self, momentum, time):
        ref_phase = self.get_phi_eff(momentum)
        phi = time*self.omega-ref_phase+self.phi_s*2.*math.pi
        while phi > math.pi*2.:
            phi -= 2.*math.pi
        while phi < 0.:
            phi += 2.*math.pi
        return phi

    def phase_to_time(self, momentum, phi):
        ref_phase = self.get_phi_eff(momentum)
        relative_phase = phi+ref_phase-self.phi_s*2.*math.pi
        while relative_phase > 0.:
            relative_phase -= 2.*math.pi
        while relative_phase < 0.:
            relative_phase += 2.*math.pi
        relative_time = relative_phase/self.omega
        print("Phase to time", ref_phase, relative_phase, relative_time)
        return relative_time

    def evolve_rf(self, momentum, time):
        out_list = []
        tof0 = self.tof(self.p0)
        print("Energy Kick", end=' ')
        for name in ["ke_in", "RF period", "time", "time0", "dphi/2pi", "delta_E", "ke_out"]:
            print(name.rjust(10), end=' ')
        print()
        for turn in range(self.n_turns):
            theta = 0.
            print("Turn", turn)
            delta_energy = 0.
            for rf_station in range(0, self.n_stations):
                time = self.time_kick(momentum, time, rf_station)
                momentum = self.energy_kick(momentum, time, rf_station)
                #delta_energy += self.delta_energy(momentum, time, rf_station)
                print("    Station", rf_station, "t:", time, "p:", momentum)
            #energy = (momentum**2+self.mass**2)**0.5+delta_energy
            #momentum = (energy**2-self.mass**2)**0.5
            time = self.time_kick(momentum, time, self.n_stations)
            out_list.append([momentum, time])
            if momentum > self.p1:
                break
        return out_list

    def plot_acceleration(self, out_list, line_color = 1, canvas = None, min_e=0., max_e=None):
        energy_list = [(item[0]**2+self.mass**2)**0.5 -self.mass for item in out_list]
        phase_list = [self.time_to_phase(item[0], item[1])/math.pi/2. for item in out_list]
        if max_e == None:
            max_e = ((self.p1**2+self.mass**2)**0.5-self.mass)*1.1
        hist, graph = xboa.common.make_root_graph("acceleration",
                                                  phase_list, "#phi [rad/2#pi]",
                                                  energy_list, "KE [MeV]", 
                                                  xmin=0.0, xmax=1.,
                                                  ymin=min_e, ymax=max_e,
                                                  sort = False)
        if canvas == None:
            canvas = xboa.common.make_root_canvas("acceleration")
            canvas.Draw()
            hist.Draw()
        canvas.cd()
        graph.SetLineColor(line_color)
        graph.SetMarkerColor(line_color)
        graph.SetMarkerStyle(7)
        graph.Draw("SAME P")
        canvas.Update()
        canvas.Print(self.plot_dir+"acceleration.pdf")
        return canvas

    def flooring(self, time):
        frequency = self.omega/2/math.pi
        phase = time*frequency
        phase = phase-math.floor(phase)
        time = phase/frequency
        return time

    def plot_snapshot(self, out_list_of_lists, index):
        energy_list = [(item[index][0]**2+self.mass**2)**0.5 -self.mass for item in out_list_of_lists]
        time_list = [self.flooring(item[index][1]) for item in out_list_of_lists]
        
        #phase_list = [phase for phase in phase_list]
        max_e = ((self.p1**2+self.mass**2)**0.5-self.mass)*1.1
        canvas = xboa.common.make_root_canvas("snapshot "+str(index))
        canvas.Draw()
        hist, graph = xboa.common.make_root_graph("snapshot "+str(index), time_list, "t [ns]", energy_list, "KE [MeV]", sort = False)
        hist.Draw()
        graph.SetMarkerStyle(20)
        graph.Draw("SAME P")
        canvas.Update()
        canvas.Print(self.plot_dir+"snapshot_"+str(index)+".pdf")
        return canvas

    @classmethod
    def ke(cls, p):
        return (p**2+cls.mass**2)**0.5-cls.mass

    def plot_bucket(self, out_list, ref_list, line_color = 1, canvas = None):
        energy_list = [self.ke(item[0]) - self.ke(ref_list[i][0]) for i, item in enumerate(out_list)]
        phase_list = [item[1]-ref_list[i][1] for i, item in enumerate(out_list)]
        hist, graph = xboa.common.make_root_graph("bucket", phase_list, "#phi [rad/2#pi]", energy_list, "KE [MeV]", xmin=-1, xmax=1., ymin=-0.2, ymax=0.2, sort = False)
        if canvas == None:
            canvas = xboa.common.make_root_canvas("bucket")
            canvas.Draw()
            hist.Draw()
        graph.SetLineColor(line_color)
        graph.SetMarkerColor(line_color)
        graph.SetMarkerStyle(7)
        graph.Draw("SAME P")
        canvas.Update()
        canvas.Print(self.plot_dir+"bucket.pdf")
        return canvas

    def plot_radial_dependence(self, line_color, station, canvas = None):
        p_step = (self.p1-self.p0)/10000
        momentum = [self.p0+i*p_step for i in range(10001)]
        radius = [self.radius(p) for p in momentum]
        voltage = [self.get_rf(p, 50., station)[1] for p in momentum]
        hist, graph = xboa.common.make_root_graph("r vs v", radius, "r [mm]", voltage, "V [MV]")
        #hist, graph = xboa.common.make_root_graph("r vs v", momentum, "p [MeV/c]", voltage, "V [MV]")
        if canvas == None:
            canvas = xboa.common.make_root_canvas("r vs v")
            hist.Draw()
        graph.SetLineColor(line_color)
        graph.Draw("SAME L")
        canvas.Update()
        for fmt in ["png", "pdf", "root"]:
            canvas.Print(self.plot_dir+"/radial_voltage_dependence."+fmt)
        fout = open(self.output_dir+"/radial_voltage_dependence_"+str(station)+".out", "w")
        for i, r in enumerate(radius):
            p = momentum[i]
            v = voltage[i]*10. # kV
            print(format(r-self.r0, "15.8g"), format(v, "15.8g"), file=fout)
        return canvas, hist, graph

    def plot_waveform(self, momentum, line_color, station = None, axes = False, canvas = None):
        if station == None:
            rf_stations = list(range(self.n_stations))
        else:
            rf_stations = [station]
        n_points = 101
        time = [i/(n_points-1.)*2.*math.pi/self.omega for i in range(0, n_points, 1)]
        voltage = [0. for i in range(len(time))]
        phi_eff, ref_voltage = self.get_phi_eff(momentum), 0.
        while phi_eff > 2.*math.pi:
            phi_eff -= 2.*math.pi
        while phi_eff < 0.:
            phi_eff += 2.*math.pi
        for a_station in rf_stations:
            omega, v, phase = self.get_rf(momentum, 0., a_station)
            if station != None:
                print("    plot_waveform station:", a_station, "omega:", omega, "v:", v, "phi_i:", phase)
            for i, t in enumerate(time):
                voltage[i] += v*math.sin(t*omega+2.*math.pi*phase)
            ref_voltage += v*math.sin(phi_eff+2.*math.pi*phase)
        if station == None:
            print("Ref voltage", ref_voltage)
        hist, graph = xboa.common.make_root_graph("v", time, "t [ns]", voltage, "V [MV]")
        if canvas == None:
            canvas = xboa.common.make_root_canvas("v")
        if axes == True:
            hist.Draw()
        graph.SetLineColor(line_color)
        graph.Draw("L")
        ref_hist, ref_graph = xboa.common.make_root_graph("v", [phi_eff/omega], "t [ns]", [ref_voltage], "v [MV]")
        ref_graph.SetMarkerStyle(24)
        ref_graph.SetMarkerColor(line_color)
        ref_graph.Draw("SAME P")
        canvas.Update()
        return canvas, hist, graph

    mass = xboa.common.pdg_pid_to_mass[2212]
    c_light = xboa.common.constants["c_light"]

def get_toy_fets_ring():
    """Parameters version 0.5 N=15 FETS ring"""
    toy = ToyLongitudinal()
    toy.r0 = 4037.
    toy.p0 = 75.091
    toy.k = 7.1
    toy.n_stations = 4
    toy.v_eff = 5e-2
    toy.phi_s = 0.1
    toy.n_turns = 2000
    toy.p1 = 239.158
    toy.scallop_factor = 1073.47/toy.tof(toy.p0)
    toy.omega = 2.*math.pi*0.01/toy.tof(toy.p0)
    toy.cavity_phase = [0., 0.25, 0.5, 0.75] # fixed actual phase of cavity
    toy.azimuthal_angle = [0.]+[0., 0.25, 0.5, 0.75]+[1.] # position of cavity
    toy.setup_lookup("data/find_closed_orbit.out", n_cells=15)
    print("Built ring with frequency", toy.omega/2./math.pi, "scallop factor", toy.scallop_factor)

    toy.plot_dir = "plots/fets_ring/"

    toy.print_radius()
    toy.setup_phi_eff()
    return toy

def get_toy_isis2_ring():
    """Parameters version 0.5 N=15 FETS ring"""
    toy = ToyLongitudinal()
    toy.r0 = 24000.
    toy.p0 = 954.3
    toy.k = 20.288
    toy.n_stations = 4
    toy.v_eff = 2
    toy.phi_s = 0.2
    toy.n_turns = 12000
    toy.p1 = 1921.4

    toy.omega = 2.*math.pi*0.01/toy.tof(toy.p0)
    toy.cavity_phase = [0., 0.25, 0.5, 0.75] # fixed actual phase of cavity
    toy.azimuthal_angle = [0.]+[0., 0.25, 0.5, 0.75]+[1.] # position of cavity

    toy.plot_dir = "plots/isis_2/"

    toy.print_radius()
    toy.setup_phi_eff()
    return toy


def math_test():
    rf_period = 128.
    time = 50.
    vref_prime = 0.5
    vref = 0.5
    dt = 1e-2
    keys = ["v0", "v1", "c2-s2", "v0*s0", "v1*c0", "v_tot", "v'_tot", "dv"]
    for key in keys:
	print(str(key).rjust(8), end=' ')
    print()

    for i in range(0, 100):
        time = rf_period*i/100.
        cos = math.cos(2.*math.pi*time/rf_period)
        sin = math.sin(2.*math.pi*time/rf_period)
        v0 = (vref*sin+vref_prime*cos)
        v1 = (vref*cos-vref_prime*sin)
        v_tot = v0*sin+v1*cos
        v_prime = v0*cos-v1*sin
        cos = math.cos(2.*math.pi*(time+dt
                            )/rf_period)
        sin = math.sin(2.*math.pi*(time+dt)/rf_period)
        dv_tot = (v0*sin+v1*cos - v_tot)/dt

        values = [v0, v1, cos**2 - sin**2, v0*sin, v1*cos, v_tot, v_prime, dv_tot]
        for value in values:
            print(str(round(value, 4)).rjust(8), end=' ')
        print()

def plot_cavities(toy):
    momenta = [toy.p0, (toy.p0+toy.p1)/2., toy.p1] #[(toy.p1 - toy.p0)*i/2.+toy.p0 for i in range(3)]
    colors = [ROOT.kGreen+3, ROOT.kBlue, ROOT.kRed+2, ROOT.kMagenta+2]
    parent_canvas = xboa.common.make_root_canvas("v")
    parent_canvas.Divide(1, len(momenta), 0, 0)
    for i, momentum in enumerate(momenta):
        canvas = parent_canvas.cd(i+1)
        canvas, hist, graph = toy.plot_waveform(momentum, ROOT.kBlack, None, True, canvas)
        for station in range(toy.n_stations):
            toy.plot_waveform(momentum, colors[station], station, False, canvas)
        parent_canvas.Print(toy.plot_dir+"waveform.pdf")
    parent_canvas.cd()
    parent_canvas.Update()
    canvas = None
    for station in range(toy.n_stations):
        canvas, hist, graph = toy.plot_radial_dependence(colors[station], station, canvas)
        
def gen(a_list):
    for item in a_list:
        yield item

def main():
    #toy = get_toy_fets_ring()
    toy = get_toy_fets_ring()
    out_list_of_lists = []
    for i in range(10):
        p = 14.45+i/100.
    plot_cavities(toy)
    canvas = None
    for i in range(10):
        p = toy.p0+i
        time_in = toy.get_phi_eff(p)/toy.omega
        #time = toy.phase_to_time(p, phase_in)
        phase_out = toy.time_to_phase(p, time_in)
        print("Phase test", time_in, phase_out)
    print(math.tan(toy.phi_s*2*math.pi), toy.omega*toy.tof(toy.p0), toy.omega*toy.tof(toy.p0)*math.tan(toy.phi_s*2*math.pi)-1)
    input("Ready to track - continue?")
    colours = iter([ROOT.kGreen-9, ROOT.kGreen, ROOT.kGreen+2,
                    ROOT.kBlue-9, ROOT.kBlue, ROOT.kBlue+2,
                    ROOT.kMagenta-9, ROOT.kMagenta-2, ROOT.kMagenta,
                    ROOT.kRed-0, ROOT.kRed, ROOT.kRed+2])
    for phi, momentum in [
        #(toy.phi_s*2.*math.pi, toy.p0), 
        (0, toy.p0-5.0), (0, toy.p0), (0, toy.p0+5.0),
        (-0.2, toy.p0-5.0), (-0.2, toy.p0), (-0.2, toy.p0+5.0),
        (0.2, toy.p0-5.0), (0.2, toy.p0), (0.2, toy.p0+5.0),
        (0.35, toy.p0), (0.5, toy.p0), (0.65, toy.p0),
        ]:
        phi = (phi+toy.phi_s)*2.*math.pi
        time = toy.phase_to_time(momentum, phi)
        out_list = toy.evolve_rf(momentum, time)
        out_list_of_lists.append(out_list)
        canvas = toy.plot_acceleration(out_list, next(colours), canvas)
    for i, out_list in enumerate(out_list_of_lists):
        print("Event", i, "made n_turns:", len(out_list), "Start:", out_list[0][1], "Finish:", out_list[-1][1])
    print("RF Period was", 2.*math.pi/toy.omega)
    canvas = None
    n_turns_list = [len(out_list) for out_list in out_list_of_lists]
    min_n_turns = min(n_turns_list)
    for i, out_list in enumerate(out_list_of_lists[1:]):
        ref_list = out_list_of_lists[0]
        canvas = toy.plot_bucket(out_list[:min_n_turns], ref_list[:min_n_turns], i+2, canvas)
    for index in 0, -1:
        toy.plot_snapshot(out_list_of_lists, index)
    input()

# NOTES:
# The feature size goes inversely with frequency; so if we want feature size
# ~ mm then we need frequency < MHz; feature size ~ cm then frequency < 0.1
# MHz. The gap height presumably has to be ~ feature size to get desired
# voltage profile, so *either*:
# ... reduce the frequency swing
# ... reduce the frequency (what implications?)
# ... accept small gap height
# Small vertical and horizontal acceptance right now!
#
# There appears to be a strong bunching effect. This is interesting; why does
# it appear? Is it real or a bug? It's a nice feature if it is real

if __name__ == "__main__":
    main()


