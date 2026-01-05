"""
Script to find the RF set up; drives find closed orbit
"""

import ctypes
import time
import os
import sys
import copy
import json
import math
import warnings
warnings.filterwarnings("ignore")

import numpy
import ROOT
import platypus

import xboa.common
from xboa.hit import Hit
sys.path.insert(1, "scripts")
from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.polynomial_fitter import PolynomialFitter


class Optimisation(object):
    def __init__(self, config):
        self.config = config
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.optimisation["run_dir"])
        self.x_err = 1.
        self.p_err = 1.
        self.fixed = {}
        self.first_tracking = True
        self.subs = {}
        self.iteration = 0
        self.tracking_result = []
        self.score = None
        self.target_hit = None
        self.overrides = {}
        self.store_index = 1
        self.opt_i = None
        self.parameters = []
        self.tracking = None
        DecoupledTransferMatrix.det_tolerance = 1.

    def setup_optimiser(self):
        self.setup(self.config.optimisation)
        self.optimisation_stages = []
        for optimisation_stage in self.config.optimisation["staged_optimisation"]:
            parameter_list = []
            for parameter_config in optimisation_stage["parameters"]:
                parameter_list.append(Parameter.setup(parameter_config))
            optimiser = Optimiser()
            optimiser.setup(optimisation_stage["optimiser"])
            optimiser.parameter_list = parameter_list
            optimiser.score_list = Score.setup_scores(optimisation_stage["score"])
            self.optimisation_stages.append(optimiser)

    def optimise(self):
        for optimiser in self.optimisation_stages:
            optimiser.run_optimisation()

    def store_data(self):
        outname = self.get_filename_root()+"_"+f'{self.store_index:03}'+".out"
        tmpname = self.get_filename_root()+".tmp"
        print("Moving data from", tmpname, "to", outname)
        os.rename(tmpname, outname)
        self.store_index += 1

    @classmethod
    def setup(cls, config):
        pass

    def get_filename_root(self):
        fname = self.config.run_control["output_dir"]+"/"
        fname += self.config.optimisation["output_file"]
        return fname

    def save_state(self, suffix, append):
        saved_score_list = []
        for score_list in self.score_dict.values():
            for score in score_list:
                saved_score_list.append(score.save_score())

        state = {
            "target_orbit":self.optimisation["target_orbit"],
            "parameters":[par.to_dict() for par in self.parameters],
            "tracking":copy.deepcopy(self.tracking_result),
            "n_iterations":self.iteration,
            "target_hit":copy.deepcopy(self.target_hit),
            "score":self.score,
            "subs":self.overrides,
            "optimisation_stage":self.opt_i,
            "score_list":saved_score_list,
        }
        fname = self.get_filename_root()+"."+suffix
        fout = open(fname, "a")
        print(json.dumps(state), file=fout)
 
    def update_parameters_from_minuit(self):
        for parameter in self.parameters:
            parameter.update_from_minuit(self.minuit)

    def penalty_factor(self, hit_list):
        target_n_hits = self.config.optimisation["target_n_hits"]
        n_hits = len(hit_list)
        penalty_factor = self.config.optimisation["penalty_factor"]
        if n_hits < target_n_hits:
            penalty = penalty_factor**(target_n_hits-n_hits)
            print("Applying penalty", penalty, "for n_hits", n_hits, "<", target_n_hits)
        else:
            penalty = 1
        return penalty

    def get_score(self, hit_list_of_lists):
        penalty = self.penalty_factor(hit_list_of_lists[0])
        normalisation = penalty/sum([len(score_list) for score_list in self.score_dict.values()])
        total_score = []
        for score_type, score_list in self.score_dict.items():
            my_scores = []
            for score in score_list:
                a_score = score.get_score(hit_list_of_lists)
                if a_score:
                    my_scores.append(a_score)
                score.print()
            # sum over all the scores
            if len(my_scores):
                # normalise to the number of scores registered
                this_score = numpy.array(my_scores)*1.0/len(my_scores)
                this_score = numpy.sum(this_score, axis=0).tolist()
                total_score += this_score
        return total_score

    def minuit_function(self, nvar=None, parameters=None, score=ctypes.c_float(0.0), jacobian=None, err=None):
        self.update_parameters_from_minuit()
        # a bit stupid, to get the interface right we convert from minuit to 
        # dict to list to dict
        score_list = self.moo_function()
        try:
            score.value =  sum(self.score)
        except TypeError:
            score.value = self.score

    def moo_function(self):
        self.iteration += 1
        if self.iteration > self.max_iterations:
            raise StopIteration("Hit maximum iteration")
        print("Running iteration", self.iteration)
        delta = time.time()
        hit_list_of_lists = self.track()
        delta = time.time()-delta
        score_list = self.get_score(hit_list_of_lists)
        self.score = sum(score_list)

        print("Tracked for time", delta, "s")
        print("score          ", format(self.score, "12.4g"), flush=True)
        print()
        self.save_state("tmp", True)
        return score_list

    def setup_subs(self, hit_list):
        subs = self.config.substitution_list[0]
        if self.first_tracking:
            for key in sorted(subs.keys()):
                print(utilities.sub_to_name(key), subs[key], end=' ')
            self.first_tracking = False
        print("Loading parameters")
        for par in self.parameters:
            par.update_to_subs(self.overrides)
            print("    ", par.name, par.current_value)
        utilities.do_lattice(self.config, self.subs, self.overrides, hit_list, self.tracking)

    def cuts(self, hit_list):
        min_delta = self.config.optimisation["min_time_delta"]
        index = 0
        while index+1 < len(hit_list):
            if hit_list[index+1]["t"] - hit_list[index]["t"] < min_delta:
                del hit_list[index+1]
            elif hit_list[index+1]["station"] == hit_list[index]["station"]:
                del hit_list[index+1]
            else:
                index += 1
        return hit_list

    def track(self):
        hit_list = [delta for delta in self.generate_delta()]
        if False:
            for hit in hit_list:
                print(hit)
        for values in self.score_dict.values():
            if len(values) and values[0].requires_transfer_matrix():
                return self.track_some(hit_list)[1:]
        return self.track_some(hit_list[0:1])

    def generate_delta(self):
        var_list = self.config.optimisation["vars"]
        delta_dict = self.config.optimisation["deltas"]
        co = self.config.optimisation["closed_orbit"]
        i_list = [i for i in range(len(co))]
        co_dict = dict([(var_list[i], co[i]) for i in i_list])
        yield copy.deepcopy(co_dict)
        yield copy.deepcopy(co_dict)
        for d in [-1, +1]:
            for key in var_list:
                psv = copy.deepcopy(co_dict)
                psv[key] += delta_dict[key]*d
                yield psv

    def track_some(self, psv_list):
        utilities.clear_dir(self.tmp_dir)
        os.chdir(self.tmp_dir)
        ref_probes = self.config.optimisation["ref_probe_files"]
        ref_energy = self.config.optimisation["energy"]
        # fix momentum
        self.tracking = utilities.setup_tracking(self.config, ref_probes, ref_energy)
        hit_list = []
        for psv in psv_list:
            my_hit = copy.deepcopy(self.tracking.ref)
            for var in psv:
                my_hit[var] = psv[var]
            if "x'" in self.config.optimisation["vars"]:
                my_hit.mass_shell_condition("p")
            else:
                my_hit.mass_shell_condition("pz")
            hit_list.append(my_hit)
        self.setup_subs(hit_list)
        print("Reference kinetic energy:", self.tracking.ref["kinetic_energy"])
        print("Seed kinetic energy:     ", [hit["kinetic_energy"] for hit in hit_list], flush=True)
        hit_list_of_lists = self.tracking.track_many(hit_list)
        print("Station to probe mapping:\n   ", end=' ')
        for i, fname in enumerate(self.tracking.get_name_list()):
            print("("+str(i)+",", fname+")", end=' ')
        print()
        return hit_list_of_lists

    root_algorithms = ["simplex", "migrad"]

global BUMP_FINDER
def minuit_function(nvar, parameters, score, jacobian, err):
    global BUMP_FINDER
    BUMP_FINDER.minuit_function(nvar, parameters, score, jacobian, err)


def main(config):
    optimisation = Optimisation(config)
    optimisation.setup_optimiser()
    optimisation.optimise()

co = [0.0, 0.0, 0.0, 0.0]
bump = [0.0, 1.0, 0.0, 0.0]
bz = 0.2

class TestConfig:
    run_control = {
        "output_dir":"test",
    }
    substitution_list = [{}]
    optimisation = {
        "output_file":"find_bump_parameters",
        "closed_orbit":co,
        "vars":["x", "x'", "y", "y'"],
        "deltas":{"x":1.0e-3, "x'":1e-5, "y":1.0e-3, "y'":1e-5},
        "magnet_min_field":-1.0,
        "magnet_max_field":+1.0,
        "max_iterations":1000,
        "target_score":0.001,
        "field_tolerance":1e-4,
        "amplitude_tolerance":1.,
        "tm_source":"../../baseline/closed_orbits_cache",
        "stop_on_fail":True,
        "subs_overrides":{
            "__n_turns__":1.11,
            "__do_magnet_field_maps__":False,
            "__do_bump__":True,
            "__do_foil__":False,
            "__do_rf__":False,
            "__n_particles__":3,
        },
        "final_subs_overrides":{
            "__n_turns__":0.99,
            "__do_magnet_field_maps__":True,
            "__do_bump__":True,
            "__n_particles__":3,
        },
        "staged_optimisation":[{ # get BF
                "parameters":[
                    {"name":"h bump 1", "key":"__h_bump_1_field__", "seed":0.0, "error":0.1, "fixed":False},
                    {"name":"h bump 2", "key":"__h_bump_2_field__", "seed":0.0, "error":0.1, "fixed":False},
                    {"name":"h bump 3", "key":"__h_bump_3_field__", "seed":bz,  "error":0.1, "fixed":True},
                    {"name":"h bump 4", "key":"__h_bump_4_field__", "seed":0.0, "error":0.1, "lower_limit":-1.0, "upper_limit":1.0, "fixed":True},
                    {"name":"h bump 5", "key":"__h_bump_5_field__", "seed":0.0, "error":0.1, "lower_limit":-1.0, "upper_limit":1.0, "fixed":True},
                ],
                "score":[{
                    "type":"psv",
                    "station":station,
                    "vector":[0, 0, 0, 0],
                    "tolerance":[1e-3, 1e-6, 1e-3, 1e-6], # default if no tolerance is given
                    "variables":["x", "px", "y", "py"], # default if no tolerance is given
                } for station in [0, 1, 2, 3]],
                "optimiser":{},
            },
        ],
        "target_fields":{},
        "seed_errors":[1e-4]*10,
        "ref_probe_files":["FOILPROBE_1.h5", "RINGPROBE*.h5"], # sorted alphanumerically
        "run_dir":"tmp/find_bump/",
        "energy":3.0,
        "min_time_delta":0., # minimum time between probes
        "target_n_hits":2,
        "penalty_factor":1e9, # penalty = p_f^(number of missed stations)
        "algorithm":"simplex",
    }

if __name__ == "__main__":
    main(TestConfig())
