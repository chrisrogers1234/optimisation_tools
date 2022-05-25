import os
import json
import math
import copy
import shutil
import numpy

import ROOT
import xboa.common
from xboa.hit import Hit

from optimisation_tools.utils import utilities

from optimisation_tools.fixed_frequency_rf.longitudinal_study import ToyLongitudinal

class FindFixedFrequency(object):
    # lookup microtron
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
        self.fout = open(self.output_dir+"/find_fixed_frequency.tmp", "w")
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.find_fixed_frequency["run_dir"])
        self.plot_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.find_fixed_frequency["plot_dir"])
        self.start = None
        self.end = None
        self.co_file = os.path.join(self.config.run_control["output_dir"],
                               self.config.find_fixed_frequency["co_file"])
        self.acceleration_canvas = None
        self.tof_canvas = None
        self.dx_canvas = None
        self.co_list = []

    def load_co_file(self):
        fin = open(self.co_file)
        self.co_list = []
        for line in fin.readlines():
            self.co_list.append(json.loads(line))
        
    def get_toy_fets_ring(self):
        """Parameters version 0.5 N=15 FETS ring"""
        self.toy = ToyLongitudinal()
        self.start = Hit.new_from_dict(self.co_list[0]['hits'][0])
        self.end = Hit.new_from_dict(self.co_list[-1]['hits'][0])
        self.toy.r0 = self.start['r']
        self.toy.p0 = self.start['p']
        self.toy.k = 7.1
        self.toy.n_stations = 4
        self.toy.v_eff = 5e-3
        self.toy.phi_s = 0.1
        self.toy.n_turns = 2000
        self.toy.p1 = self.end['p']
        self.toy.scallop_factor = 1073.47/self.toy.tof(self.toy.p0)
        self.toy.omega = 2.*math.pi*0.001/self.toy.tof(self.toy.p0)
        self.toy.cavity_phase = [0., 0.25, 0.5, 0.75] # fixed actual phase of cavity
        self.toy.azimuthal_angle = [0.]+[0., 0.25, 0.5, 0.75]+[1.] # position of cavity
        self.toy.setup_lookup(self.co_file, n_cells=15)
        print("Built ring with frequency", self.toy.omega/2./math.pi, \
              "scallop factor", self.toy.scallop_factor)

        self.toy.plot_dir = self.plot_dir
        self.toy.output_dir = self.tmp_dir

        self.toy.print_radius()
        self.toy.setup_phi_eff()
        canvas = None
        for station, color in [(0, ROOT.kRed), (1, ROOT.kGreen), (2, ROOT.kBlue), (3, ROOT.kMagenta)]:
            canvas, hist, graph = self.toy.plot_radial_dependence(color, station, canvas)

    def do_substitutions(self, phi_init):
        overrides = self.config.find_fixed_frequency["subs_overrides"]
        self.subs = copy.deepcopy(self.config.substitution_list[0])
        for key, value in overrides.items():
            self.subs[key] = value
        self.subs["__rf_phase__"] = phi_init
        self.subs["__rf_freq_0__"] = self.toy.omega/math.pi/2.
        xboa.common.substitute(self.config.tracking["lattice_file"],
                              self.tmp_dir+'/SectorFFAGMagnet.tmp',
                              self.subs)

    def get_hit(self):
        test_hit = Hit.new_from_dict(self.co_list[0]['hits'][0])
        #print test_hit['x'], test_hit['y'], test_hit['z']
        #radius = test_hit['x']
        #pr, pphi = test_hit['px'], test_hit['pz']
        #test_hit['x'] = radius*math.cos(phi)
        #test_hit['z'] = radius*math.sin(phi)
        #test_hit['px'] = pr*math.cos(phi)+pphi*math.sin(phi)
        #test_hit['pz'] = -pr*math.sin(phi)+pphi*math.cos(phi)
        #print "track with phi", phi, "r", (test_hit['x']**2+test_hit['z']**2)**0.5
        return test_hit
        
    def tracking(self):
        probe_files = self.config.find_fixed_frequency["probe_files"]
        phi_list = [i/10. for i in range(-5, 6)]+[-0.05, -0.01, +0.05, +0.01]
        phi_list = sorted(phi_list, key = lambda x: abs(x))
        clr_list = numpy.array([1, ROOT.kRed+2, ROOT.kRed, ROOT.kRed-9, ROOT.kBlue+2, ROOT.kBlue, ROOT.kBlue-9])
        ldots = [{'kinetic_energy':'...', 't':'...'}]
        for i, phi in enumerate(phi_list):
            self.do_substitutions(phi) # RF phase goes from 0 to 1
            test_hit = self.get_hit()
            tracking = utilities.setup_tracking(self.config, probe_files, test_hit['kinetic_energy'])
            os.chdir(self.tmp_dir)
            hit_list = tracking.track_one(test_hit)
            lattice_name = self.config.tracking["lattice_file_out"].split(".")[0]
            os.rename(lattice_name+"-trackOrbit.dat",
                      lattice_name+"-trackOrbit-"+str(i+1)+".dat")
            max_e = max([hit['kinetic_energy'] for hit in hit_list])
            print(len(hit_list), "hits with peak energy", max_e, "start and end t, E:")
            print([(hit['t'], hit['kinetic_energy']) for hit in hit_list[0:2]+ldots+hit_list[-3:-1]])
            color = clr_list[0]
            clr_list = numpy.roll(clr_list, 1)
            self.plot_acceleration(hit_list, phi+0.5, color)

    def plot_acceleration(self, hit_list, phi, color):
        print("Plot acceleration", hit_list[0]['t'], phi, self.toy.omega)
        out_list = [[hit['p'], hit['t']+2.*math.pi*phi/self.toy.omega] for hit in hit_list]
        self.acceleration_canvas = self.toy.plot_acceleration(out_list,
                                                              color,
                                                              self.acceleration_canvas,
                                                              #min_e=2.9,
                                                              #max_e=3.1
                                                              )
        self.plot_tof(hit_list, color)
        self.plot_pos(hit_list, color)
        self.output(hit_list)

    def plot_tof(self, hit_list, color):
        e_list = [hit['kinetic_energy'] for hit in hit_list[1:]]
        dt_list = [hit['t']-hit_list[i]['t']-self.toy.tof(hit['p']) for i, hit in enumerate(hit_list[1:])]
        for i, hit in enumerate(hit_list[1:10]):
            print(hit['t'], hit_list[i]['t'], self.toy.tof(hit['p']))
        print(dt_list[0:10])
        print(e_list[0:10])
        dt_max = 0.1/self.toy.omega
        hist, graph = xboa.common.make_root_graph("e vs dt",
                                                  dt_list, 'dt [ns]',
                                                  e_list, 'energy [MeV]',
                                                  xmin=-dt_max, xmax=dt_max,
                                                  ymin=0.,
                                                  ymax=self.end['kinetic_energy'])
        if self.tof_canvas == None:
            self.tof_canvas = ROOT.TCanvas("tof", "tof")
            hist.Draw()
        self.tof_canvas.cd()
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(7)
        graph.Draw("P SAME")
        self.tof_canvas.Update()
        for fmt in ["pdf"]:
            self.tof_canvas.Print(self.plot_dir+"tof_plot."+fmt)

    def plot_pos(self, hit_list, color):
        e_list = [hit['kinetic_energy'] for hit in hit_list]
        dx_list = [hit['x']-self.toy.radius(hit['p']) for hit in hit_list]
        hist, graph = xboa.common.make_root_graph("e vs dx",
                                                  dx_list, 'dx [mm]',
                                                  e_list, 'energy [MeV]',
                                                  ymin=0.,
                                                  ymax=self.end['kinetic_energy'])
        if self.dx_canvas == None:
            self.dx_canvas = ROOT.TCanvas("dx", "dx")
            hist.Draw()
        self.dx_canvas.cd()
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(7)
        graph.Draw("P SAME")
        self.dx_canvas.Update()
        for fmt in ["pdf"]:
            self.dx_canvas.Print(self.plot_dir+"dx_plot."+fmt)

    def output(self, hit_list):
        output = {"substitutions":self.subs, "hits":[hit.dict_from_hit() for hit in hit_list]}
        print(json.dumps(output), file=self.fout)
        self.fout.flush()


    def setup_dirs(self):
        for a_dir in [self.tmp_dir, self.plot_dir]:
            try:
                shutil.rmtree(a_dir)
            except OSError:
                pass
            os.makedirs(a_dir)
        
    def find_fixed_frequency(self):
        self.load_co_file()
        self.setup_dirs()
        self.get_toy_fets_ring()
        self.tracking()
    
def main(config):
    fff = FindFixedFrequency(config)
    fff.find_fixed_frequency()
