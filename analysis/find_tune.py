"""
Script to find the tune; drives xboa DPhiTuneFinder (FFT was making large side
bands which looked non-physical)
"""

import glob
import json
import copy
import sys
import math
import os
import shutil
import ROOT
import xboa.common
from xboa.hit import Hit
from xboa.algorithms.tune import FFTTuneFinder
from xboa.algorithms.tune import DPhiTuneFinder

from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils import utilities

class Tune(object):
    def __init__(self, config):
        """
        Find the tune. 

        -probe_file_name: name of a PROBE file from OPAL output, or None. If file_name
                    is specified, use that file_name in order to calculate tunes
                    (and generate plots), otherwise generate a new one by
                    tracking.
        -closed_orbits_file_name: name of a file containing closed orbits,
                    generated by e.g. 
        """
        self.config = config
        self.var_list = ["x", "px", "y", "py"]
        self.output_dir = os.path.join(self.config.run_control["output_dir"])
        self.output_filename = os.path.join(self.output_dir,
                                        self.config.find_tune["output_file"])
        self.run_dir = os.path.join(self.output_dir,
                                    self.config.find_tune["run_dir"])
        self._load_closed_orbits()
        self.tracking = None
        self.here = os.getcwd()
        DecoupledTransferMatrix.det_tolerance = 1.
        utilities.clear_dir(self.run_dir)

    def track_decoupled(self, co_element, decoupled_psv):
        overrides = self.config.find_tune["subs_overrides"]
        tm = copy.deepcopy(co_element["tm"])
        for i, row in enumerate(tm):
            tm[i] = row[1:5]
        tm = DecoupledTransferMatrix(tm)
        coupled_psv = tm.coupled(decoupled_psv)
        print("Decoupled:", decoupled_psv)
        print("Coupled:  ", coupled_psv)
        ref_hit = Hit.new_from_dict(co_element["seed_hit"])
        test_hit = copy.deepcopy(ref_hit)
        for i, var in enumerate(self.var_list):
            print("Adding", coupled_psv[i], "to", var, "before", test_hit[var], end=' ')
            test_hit[var] += coupled_psv[i]
            print("after", test_hit[var])
        test_hit.mass_shell_condition("pz")
        os.chdir(self.run_dir)
        subs = utilities.do_lattice(self.config, co_element["substitutions"], overrides)
        self.tracking = utilities.setup_tracking(self.config,
                                                 self.config.find_tune["probe_files"],
                                                 subs["__energy__"])
        self.tracking.verbose = True
        track = self.tracking.track_one(test_hit)
        print("Found", len(track), "hits")
        os.chdir(self.here)
        decoupled_track = []
        for hit in track:
            print("    ", end=' ')
            for var in self.var_list:
                print(format(hit[var], "14.8g"), end=' ')
            print()
            coupled_psv_out = [hit[var]-ref_hit[var] for var in self.var_list]
            decoupled_psv_out = tm.decoupled(coupled_psv_out)
            decoupled_track.append(decoupled_psv_out)
        return decoupled_track

    def set_finder(self, finder, axis, psv_list):
        if axis == "u":
            finder.u = [psv[0] for psv in psv_list]
            finder.up = [psv[1] for psv in psv_list]
        else:
            finder.u = [psv[2] for psv in psv_list]
            finder.up = [psv[3] for psv in psv_list]

    def find_tune_dphi(self):
        """
        Algorithm is to just calculate the turn-by-turn phase advance; this is
        done by evolving turn-by-turn the track; calculating a matched ellipse
        by looking at tracking output; transforming the ellipse into a circle
        using LU decomposition; then calculating the angle advanced.
        """
        cwd = os.getcwd()
        fout = open(self.output_filename+".tmp", "w")
        for i, closed_orbit in enumerate(self.closed_orbits_cached):
            subs = closed_orbit["substitutions"]
            for item, key in self.config.find_tune["subs_overrides"].items():
                subs[item] = key

            print("Finding tune with", end=' ') 
            for key in sorted(subs.keys()):
                print(utilities.sub_to_name(key), subs[key], end=' ')
            print()
            tune_info = {"substitutions":subs}
            ref_psv_list = self.track_decoupled(closed_orbit, [0., 0., 0., 0.])[1:]
            for axis, not_axis in [("u", "v"), ("v", "u")]:
                finder = DPhiTuneFinder()
                u, v = 0., 0.
                if axis == "u":
                    u = self.config.find_tune["delta_1"]
                else:
                    v = self.config.find_tune["delta_2"]
                psv_list = self.track_decoupled(closed_orbit, [u, 0., v, 0.])[1:]
                self.set_finder(finder, axis, psv_list)
                try:
                    tune = finder.get_tune(subs["__n_turns__"]/10.) # tolerance
                except:
                    tune = 0.
                print('  Found', len(finder.dphi), 'dphi elements with tune', tune, "+/-", finder.tune_error)
                tune_info[axis+"_tune"] = tune
                tune_info[axis+"_tune_rms"] = finder.tune_error
                tune_info[axis+"_signal"] = list(zip(finder.u, finder.up))
                tune_info[axis+"_dphi"] = finder.dphi
                tune_info[axis+"_n_cells"] = len(finder.dphi)
                canvas, hist, graph = finder.plot_phase_space_root()
                name = os.path.join(self.output_dir, "tune_"+str(i)+"_finding-"+axis+"_"+axis+"_phase-space")
                for format in ["png"]:
                    canvas.Print(name+"."+format)
                canvas, hist, graph = finder.plot_cholesky_space_root()
                name = os.path.join(self.output_dir, "tune_"+str(i)+"_"+axis+"_cholesky-space")
                for format in ["png"]:
                    canvas.Print(name+"."+format)
                self.set_finder(finder, not_axis, psv_list)
                canvas, hist, graph = finder.plot_phase_space_root()
                name = os.path.join(self.output_dir, "tune_"+str(i)+"_finding-"+axis+"_"+not_axis+"_phase-space")
                for format in ["png"]:
                    canvas.Print(name+"."+format)
                for i, u in enumerate([]):#finder.u[:-1]):
                    up = finder.up[i]
                    dphi = finder.dphi[i]
                    t = finder.t[i]
                    u_chol = finder.point_circles[i][0]
                    up_chol = finder.point_circles[i][1]
                    phi = math.atan2(up_chol, u_chol)
                    print(str(i).ljust(4),  str(round(t, 4)).rjust(8), "...", \
                          str(round(u, 4)).rjust(8), str(round(up, 4)).rjust(8), "...", \
                          str(round(u_chol, 4)).rjust(8), str(round(up_chol, 4)).rjust(8), "...", \
                          str(round(phi, 4)).rjust(8), str(round(dphi, 4)).rjust(8))

            for key in sorted(tune_info.keys()):
                if "signal" not in key and "dphi" not in key:
                    print("   ", key, tune_info[key])
            print(json.dumps(tune_info), file=fout)
            fout.flush()
        fout.close()
        os.rename(self.output_filename+".tmp", self.output_filename)
        os.chdir(cwd)

    def _temp_dir(self):
        """Make a temporary directory for tune calculation"""
        tmp_dir = os.path.join(self.output_dir, "tmp/tune/")
        try:
            os.makedirs(tmp_dir)
        except OSError:
            pass
        os.chdir(tmp_dir)
        self.tmp_dir = "./"

    def _load_closed_orbits(self):
        """Load closed orbits from a json file"""
        filename = self.output_dir+"/"+self.config.find_closed_orbits["output_file"]
        fin = open(filename)
        closed_orbits = []
        for line in fin.readlines():
            closed_orbits += json.loads(line)
        self.closed_orbits_cached = closed_orbits

    def _reference(self, hit):
        """Generate a reference particle"""
        hit = hit.deepcopy()
        hit["x"] = 0.
        hit["px"] = 0.
        return hit

def main(config):
    tune = Tune(config)
    tune.find_tune_dphi()
    ROOT.gROOT.SetBatch(False)


if __name__ == "__main__":
    main()










