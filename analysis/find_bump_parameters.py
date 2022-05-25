"""
Script to find the RF set up; drives find closed orbit
"""

import time
import os
import sys
import copy
import json
import math

import numpy
import ROOT
import platypus

import xboa.common
from xboa.hit import Hit
sys.path.insert(1, "scripts")
from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix

class FindBumpParameters(object):
    def __init__(self, config):
        self.config = config
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.find_bump_parameters["run_dir"])
        self.x_err = 1.
        self.p_err = 1.
        self.field_names = [name for name in self.field_names_gen()]
        self.field_names = sorted(self.field_names,
                        key = lambda key: [x for x in reversed(key.split('_'))])
        self.fixed = {}
        self.first_tracking = True
        self.subs = {}
        self.iteration = 0
        self.tracking_result = []
        self.score = None
        self.target_hit = None
        self.overrides = {}
        self.bump_fields = {}
        self.optimisation_fields = []
        self.store_index = 1
        self.opt_i = None
        DecoupledTransferMatrix.det_tolerance = 1.

    def setup_minuit(self):
        self.minuit = ROOT.TMinuit(len(self.field_names))
        min_b = self.config.find_bump_parameters["magnet_min_field"]
        max_b = self.config.find_bump_parameters["magnet_max_field"]
        errors = self.config.find_bump_parameters["seed_errors"]
        fixed = self.optimisation["fix_bumps"]
        for name, field in self.optimisation["seed_fields"].items():
            self.bump_fields[name] = field
        for i, name in enumerate(self.field_names):
            self.minuit.DefineParameter(i, name,
                                        self.bump_fields[name], abs(errors[i]),
                                        min_b, max_b)
            if name in fixed:
                self.minuit.FixParameter(i)
                print("Fixed bump", name)
        self.minuit.SetFCN(self.minuit_function)

    def setup_amplitude(self, substitution_index):
        tm_source = self.config.run_control["output_dir"]+"/"+\
                    self.config.find_bump_parameters["tm_source"]
        co_source = open(tm_source)
        co_source = json.loads(co_source.readline())
        tm = co_source[substitution_index]["tm"]
        for i, row in enumerate(tm):
            tm[i] = row[1:5]
        self.tm = DecoupledTransferMatrix(tm)

    def field_names_gen(self):
        n_h_bumps = self.config.find_bump_parameters["n_h_bumps"]
        n_v_bumps = self.config.find_bump_parameters["n_v_bumps"]
        if "field_names" in self.config.find_bump_parameters:
            for name in self.config.find_bump_parameters["field_names"]:
                yield name
            return
        else:
            for i in range(n_h_bumps):
                yield "__h_bump_"+str(i+1)+"_field__"
            for i in range(n_v_bumps):
                yield "__v_bump_"+str(i+1)+"_field__"

    def store_data(self):
        outname = self.get_filename_root()+"_"+f'{self.store_index:03}'+".out"
        tmpname = self.get_filename_root()+".tmp"
        print("Moving data from", tmpname, "to", outname)
        os.rename(tmpname, outname)
        self.store_index += 1

    def bump_target_orbit(self, bump_station, bump):
        if bump_station not in self.optimisation["target_orbit"]:
            print("No bump - could not find station", bump_station)
            return
        old_target_orbit = self.optimisation["target_orbit"][bump_station]
        new_target_orbit = [old_target_orbit[i]+bump[i] for i in range(4)]+old_target_orbit[4:]
        self.optimisation["target_orbit"][bump_station] = new_target_orbit
        print("Bumped target orbit at station", bump_station,
              "from", old_target_orbit, "to", new_target_orbit)

    def find_bump_parameters(self):
        try:
            os.rename(
                self.get_filename_root()+".tmp",
                self.get_filename_root()+".old"
            )
        except OSError:
            pass
        for i, subs in enumerate(self.config.substitution_list):
            self.setup_amplitude(i)
            self.subs = subs
            self.seed_bump_fields()
            bump_index = 0
            for station, bump in self.config.find_bump_parameters["bump"]:
                for opt_i, optimisation in enumerate(self.config.find_bump_parameters["staged_optimisation"]):
                    self.opt_i = {
                        "substitution_index":i,
                        "bump_index":bump_index,
                        "optimisation_stage":opt_i
                    }
                    self.optimisation = copy.deepcopy(optimisation)
                    self.bump_target_orbit(station, bump)
                    self.do_one_optimisation()
                bump_index += 1
            self.optimisation_fields.append(self.bump_fields)

    def seed_bump_fields(self):
        if len(self.optimisation_fields) == 0:
            self.bump_fields = dict([(name, 0.) for name in self.field_names])
            return
        elif len(self.optimisation_fields) == 1:
            self.bump_fields = copy.deepcopy(self.optimisation_fields[-1])
            return
        fields_0 = self.optimisation_fields[-2]
        fields_1 = self.optimisation_fields[-1]
        self.bump_fields = copy.deepcopy(self.optimisation_fields[-1])
        for key in self.bump_fields: # assume we move linearly...
            self.bump_fields[key] += fields_1[key] - fields_0[key]

    def run_minuit(self, algorithm):
        print("setup minuit")
        self.setup_minuit()
        print("run minuit")
        try:
            self.minuit.Command(algorithm+" "+str(self.max_iterations)+" "+str(self.config.find_bump_parameters["target_score"]))
        except Exception:
            sys.excepthook(*sys.exc_info())
            print("Minuit failed")
            if self.config.find_bump_parameters["stop_on_fail"]:
                raise
        print("done minuit")

    def run_platypus(self, algorithm):
        min_b = self.config.find_bump_parameters["magnet_min_field"]
        max_b = self.config.find_bump_parameters["magnet_max_field"]
        fixed = self.optimisation["fix_bumps"]
        seed = copy.deepcopy(self.config.find_bump_parameters["seed_fields"])
        sigma = self.config.find_bump_parameters["seed_errors"][0]
        self.field_names = [name for name in self.field_names_gen()]
        max_iterations = self.config.find_bump_parameters["max_iterations"]
        n_target_fields = len(self.config.find_bump_parameters["target_fields"])


        for name in fixed:
            index = self.field_names.index(name)
            self.fixed[name] = seed[index]
            print("Fixing bump", name, "to", self.fixed[name])
            del self.field_names[index]
            del seed[index]
        n_parameters = len(self.field_names)
        n_variables = 6+n_target_fields
        problem = platypus.Problem(n_parameters, n_variables)
        problem.types[:] = platypus.Real(min_b, max_b)
        problem.function = self.moo_function
        algorithm = platypus.CMAES(problem)
        algorithm.initial_search_point = seed
        algorithm.sigma = sigma # note this is a single value
        try:
            algorithm.run(max_iterations)
        except Exception:
            sys.excepthook(*sys.exc_info())
            print("Platypus failed")

    def do_one_optimisation(self):
        print("Doing optimisation")
        self.overrides = self.config.find_bump_parameters["subs_overrides"]
        self.max_iterations = self.config.find_bump_parameters["max_iterations"]
        self.iteration = 0
        algorithm = self.config.find_bump_parameters["algorithm"]
        if str(algorithm).lower() in self.root_algorithms:
            self.run_minuit(algorithm)
        elif algorithm == "nsga2":
            self.run_platypus(algorithm)
        else:
            raise RuntimeError("Did not recognise algorithm "+str(algorithm))
        print("Finished optimisation")
        self.overrides = self.config.find_bump_parameters["final_subs_overrides"]
        if self.overrides != None:
            print("Doing final tracking")
            self.track_one(self.bump_fields)
            print("Done final tracking with overrides = {")
            for key in sorted(self.overrides.keys()):
                print("   '"+key+"' :", self.overrides[key], ",")
            print("}")
        self.store_data()
        print("End of optimisation loop\n\n")

    def get_filename_root(self):
        fname = self.config.run_control["output_dir"]+"/"
        fname += self.config.find_bump_parameters["output_file"]
        return fname

    def save_state(self, suffix, append):
        state = {
            "target_orbit":self.optimisation["target_orbit"],
            "bump_fields":self.bump_fields,
            "tracking":copy.deepcopy(self.tracking_result),
            "n_iterations":self.iteration,
            "target_hit":copy.deepcopy(self.target_hit),
            "score":self.score,
            "subs":self.overrides,
            "optimisation_stage":self.opt_i,
        }
        fname = self.get_filename_root()+"."+suffix
        fout = open(fname, "a")
        print(json.dumps(state), file=fout)
 
    def get_fields_from_minuit(self):
        fields = {}
        for i, name in enumerate(self.field_names):
            var = ROOT.Double()
            err = ROOT.Double()
            self.minuit.GetParameter(i, var, err)
            fields[name] = var
        return fields
 
    def get_fields_from_parameters(self, parameters_list):
        fields = copy.deepcopy(self.fixed)
        for i, name in enumerate(self.field_names):
            par = parameters_list[i]
            fields[name] = par
        self.bump_fields = fields
        return fields

    def get_amplitude(self, hit):
        """
        Oh what a beautiful morning
        I have put in place the machinery, but I dont *quite* know what I am
        trying to optimise for... amplitude of the H- injection relative to the
        proton closed orbit; where we move the proton closed orbit and leave the
        H- injecting to a fixed point; TM at the foil needs to be evaluated.
        
        So I think I need to do:
            1. calculate 50 mm offset bumper settings
            2. calculate TM using the bumper settings (and e.g. CO finder)
            3. scan amplitude settings relative to bumper settings
        """
        return 0., 0.

    def get_n_hits(self, hit_list, stations):
        target_n_hits = self.config.find_bump_parameters["target_n_hits"]
        hit_list = [hit for hit in hit_list if hit["station"] in stations]
        n_hits = len(hit_list)
        penalty_factor = self.config.find_bump_parameters["penalty_factor"]
        denominator = n_hits
        if denominator == 0:
            denominator = 1
        penalty = penalty_factor**(target_n_hits-n_hits)
        print("Found ", n_hits, "usable hits.",
              "Target", target_n_hits, "hits.")
        if penalty <= 1.0000001:
            penalty = 1.
        else:
            print("Penalty", penalty, "will be applied to all scores")
        return n_hits, denominator, penalty

    def get_score(self, hit_list):
        [xs, pxs, ys, pys] = self.config.find_bump_parameters["vars"]
        target_co = self.optimisation["target_orbit"]
        x_score, px_score, y_score, py_score, au_score, av_score = 0., 0., 0., 0., 0., 0.
        n_hits, denominator, penalty = self.get_n_hits(hit_list, target_co.keys())
        print("station    "+xs+"              "+pxs+"             "+ys+"              "+pys+"         | t           |",
              "        |      r_x     r_px     r_y      r_py | tolerances |")
        psv_tolerance = self.optimisation["psv_tolerance"]
        for i, hit in enumerate(hit_list):
            print(format(hit["station"], "6"),
                  format(hit[xs], "14.9g"), format(hit[pxs], "14.9g"),
                  format(hit[ys], "14.9g"), format(hit[pys], "14.9g"),
                  "|", format(hit["t"], "12.5g"), end='| ')
            if hit["station"] in target_co.keys():
                key = hit["station"]
                r_x  = hit[xs] - target_co[key][0]
                r_px = hit[pxs] - target_co[key][1]
                r_y  = hit[ys] - target_co[key][2]
                r_py = hit[pys] - target_co[key][3]
                print("orbit   |",
                      format(r_x, "8.3g"),  format(r_px, "8.3g"),
                      format(r_y, "8.3g"),  format(r_py, "8.3g"), end="")
                self.target_hit = i
                tol = copy.deepcopy(psv_tolerance)
                for j in range(4):
                    if j+4 < len(target_co[key]):
                        tol[j] = target_co[key][j+4] 
                print(" | tol |", format(tol[0], "8.3g"), format(tol[1], "8.3g"), format(tol[2], "8.3g"), format(tol[3], "8.3g"))
            else:
                print("ignored |")
                continue
            x_score += (r_x/tol[0])**2./denominator
            px_score += (r_px/tol[1])**2./denominator
            y_score += (r_y/tol[2])**2./denominator
            py_score += (r_py/tol[3])**2./denominator
        x_score = x_score*penalty
        px_score = px_score*penalty
        y_score = y_score*penalty
        py_score = py_score*penalty
        return x_score, px_score, y_score, py_score, 0, 0

    def minuit_function(self, nvar, parameters, score, jacobian, err):
        fields = self.get_fields_from_minuit()
        # a bit stupid, to get the interface right we convert from minuit to 
        # dict to list to dict
        parameters = [fields[name] for name in self.field_names]
        score_list = self.moo_function(parameters)
        score[0] =  self.score

    def moo_function(self, parameters):
        self.iteration += 1
        if self.iteration > self.max_iterations:
            raise StopIteration("Hit maximum iteration")
        a_scale = self.config.find_bump_parameters["amplitude_tolerance"]
        b_scale = self.config.find_bump_parameters["field_tolerance"]
        target_fields = self.config.find_bump_parameters["target_fields"]

        print("Running iteration", self.iteration)
        delta = time.time()
        fields = self.get_fields_from_parameters(parameters)
        hit_list = self.track_one(fields)
        delta = time.time()-delta
        x_score, px_score, z_score, pz_score, au_score, av_score = self.get_score(hit_list)
        b_score = [(fields[key] - target_fields[key])**2/b_scale**2 for key in target_fields ]
        print("Bump fields:")
        print("   ", [fields[key] for key in sorted(fields)])
        for key in sorted(fields.keys(),
                          key = lambda key: [x for x in reversed(key.split('_'))] ):
            value = fields[key]
            a_out = self.config.run_control["default_text"]
            if key in self.optimisation["fix_bumps"]:
                a_in=self.config.run_control["faint_text"]
            else:
                a_in=a_out
            print(a_in, key.replace("_", " "), str(value), a_out)
        score_list = [x_score,
               px_score,
               z_score,
               pz_score,
               au_score/a_scale**2,
               av_score/a_scale**2]+b_score
        self.score = sum(score_list)

        print("Tracked for time", delta, "s")
        print("Residuals are RMS*(penalty due to missing hits)")
        print("    x RMS residual ", format(x_score**0.5, "12.4g"))
        print("    px RMS residual", format(px_score**0.5, "12.4g"))
        print("    z RMS residual ", format(z_score**0.5, "12.4g"))
        print("    pz RMS residual", format(pz_score**0.5, "12.4g"))
        print("    b score        ", format(sum(b_score)**0.5*b_scale, "12.4g"))
        print("score          ", format(self.score, "12.4g"), flush=True)
        print()
        self.save_state("tmp", True)
        return score_list

    def setup_subs(self, fields):
        subs = self.config.substitution_list[0]
        if self.first_tracking:
            for key in sorted(subs.keys()):
                print(utilities.sub_to_name(key), subs[key], end=' ')
            self.first_tracking = False
        self.overrides.update(fields)
        utilities.do_lattice(self.config, self.subs, self.overrides)

    def cuts(self, hit_list):
        min_delta = self.config.find_bump_parameters["min_time_delta"]
        index = 0
        while index+1 < len(hit_list):
            if hit_list[index+1]["t"] - hit_list[index]["t"] < min_delta:
                del hit_list[index+1]
            elif hit_list[index+1]["station"] == hit_list[index]["station"]:
                del hit_list[index+1]
            else:
                index += 1
        return hit_list

    def track_one(self, fields):
        utilities.clear_dir(self.tmp_dir)
        os.chdir(self.tmp_dir)
        ref_probes = self.config.find_bump_parameters["ref_probe_files"]
        #bump_probes = [self.config.find_bump_parameters["bump_probe_file"]]
        ref_energy = self.config.find_bump_parameters["energy"]
        self.setup_subs(fields)
        tracking = utilities.setup_tracking(self.config, ref_probes, ref_energy)
        test_hit = tracking.ref.deepcopy()
        closed_orbit = self.config.find_bump_parameters["closed_orbit"]
        for i, var in enumerate(self.config.find_bump_parameters["vars"]):
            test_hit[var] = closed_orbit[i]
        # fix momentum
        test_hit["pz"] = (tracking.ref["p"]**2-test_hit["px"]**2)**0.5
        print("Reference kinetic energy:", tracking.ref["kinetic_energy"])
        print("Seed kinetic energy:     ", test_hit["kinetic_energy"], flush=True)
        hit_list = tracking.track_many([test_hit]*3)[0]
        print("Station to probe mapping:\n   ", end=' ')
        for i, fname in enumerate(tracking.get_name_list()):
            print("("+str(i)+",", fname+")", end=' ')
        print()
        hit_list = self.cuts(hit_list)
        self.tracking_result = [[hit["station"], hit["x"], hit["px"], hit["y"], hit["py"], hit["t"]] for hit in hit_list]
        return hit_list

    root_algorithms = ["simplex", "migrad"]

def main(config):
    find_bump = FindBumpParameters(config)
    find_bump.find_bump_parameters()
    if __name__ == "__main__":
        input()
