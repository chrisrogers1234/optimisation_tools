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


class Score(object):
    def __init__(self):
        pass

    def get_score(self):
        pass

    def print(self):
        pass

    def requires_transfer_matrix(self):
        return False

    def save_score(self):
        return {}

    @classmethod
    def number_of_scores(cls):
        return 0

    @classmethod
    def setup(cls, config, optimisation):
        return []

    @classmethod
    def setup_scores(cls, config, optimisation):
        score_dict = {}
        score_dict["PSVScore"] = PSVScore.setup(config, optimisation)
        score_dict["TuneScore"] = TuneScore.setup(config, optimisation)
        score_dict = dict([(key, value) for key, value in score_dict.items() if len(value)])
        return score_dict

class PSVScore(Score):
    def __init__(self, optimisation_def, config, optimisation):
        super(Score).__init__()
        self.vars = config.find_bump_parameters["vars"]
        self.print_vars = self.vars+["t", "kinetic_energy"]
        self.station = optimisation_def[0]
        self.print_r = [0 for var in self.print_vars]
        parameters = optimisation_def[1]
        if parameters:
            self.target_co = parameters[0:4]
            if len(parameters) > 4:
                self.tol = parameters[4:8]
            else:
                self.tol = optimisation["psv_tolerance"]
        else:
            self.target_co = None
        self.i_list = [i for i in range(len(self.vars))]
        self.fmt = "14.9g"

    @classmethod
    def number_of_scores(cls):
        return 4

    def requires_transfer_matrix(self):
        return False

    def get_score(self, hit_list_of_lists):
        hit_list = hit_list_of_lists[0]
        [xs, pxs, ys, pys] = self.vars
        x_score, px_score, y_score, py_score = 0., 0., 0., 0.
        #n_hits, denominator, penalty = self.get_n_hits(hit_list, target_co.keys())
        self.score = None
        for i, hit in enumerate(hit_list):
            if hit["station"] == self.station:
                key = hit["station"]
                self.print_r = [hit[var] for var in self.print_vars]
                if not self.target_co:
                    continue
                self.r = [hit[self.vars[i]] - self.target_co[i] for i in self.i_list]
                self.score = [(self.r[i]/self.tol[i])**2 for i in self.i_list]
                break
        return self.score

    def print(self):
        print("Orbit", self.station, end=" | ")
        for r in self.print_r:
            print(format(r, self.fmt), end=" ")
        print(" | ", end="")
        if self.score == None:
            print("Ignored")
        else:
            for s in self.score:
                print(format(s, self.fmt), end=" ")
            print("|")

    @classmethod
    def setup(cls, config, optimisation):
        if "target_orbit" not in optimisation: 
            return []
        score_list = []
        for item in sorted(optimisation["target_orbit"].items()):
            score_list.append(PSVScore(item, config, optimisation))
        return score_list

    def save_score(self):
        output = {
            "score_type":"orbit",
            "station":self.station,
            "target":self.target_co,
            "orbit":self.print_r,
            "weighted_orbit":self.score
        }
        return output


class TuneScore(Score):
    def __init__(self, optimisation_def, config, optimisation):
        super(Score).__init__()
        self.vars = config.find_bump_parameters["vars"]
        self.station = optimisation_def[0]
        parameters = optimisation_def[1]
        self.target_tune = None
        self.tune_tolerance = None
        if parameters:
            self.target_tune = parameters[0:2]
            self.tune_tolerance = parameters[2:4]
        self.i_list = [i for i in range(len(self.vars))]
        self.n_tunes = int(len(self.vars)/2)
        self.tunes = [0 for i in range(self.n_tunes)]
        self.orbit = [0 for var in self.vars]
        self.tm = numpy.array([[0 for var in self.vars] for var in self.vars])
        self.fmt = "14.9g"

    def requires_transfer_matrix(self):
        return True

    @classmethod
    def number_of_scores(cls):
        return 2

    def get_station_hits(self, hit_list_of_lists):
        source_hits = []
        station_hits = []
        for hit_list in hit_list_of_lists:
            source_hits.append(hit_list[0])
            for hit in hit_list:
                if hit["station"] == self.station:
                    station_hits.append(hit)
                    break
        return source_hits, station_hits

    def get_psv_list(self, hit_list):
        psv_list = []
        ref = hit_list[0]
        for hit in hit_list[1:]:
            psv_list.append([hit[var]-ref[var] for var in self.vars])
        return psv_list

    def get_tm(self, source_list, station_list):
        psv_in = self.get_psv_list(source_list)
        psv_out = self.get_psv_list(station_list)
        fitter = PolynomialFitter(len(self.vars))
        transfer_map = fitter.fit_transfer_map(psv_in, psv_out)
        transfer_map = numpy.array(transfer_map)[:, 1:]
        if False:
            print("IN")
            print(numpy.array(psv_in))
            print("OUT")
            print(numpy.array(psv_out))
            print("Transfer map")#, numpy.linalg.det(numpy.array(transfer_map)))
            print(transfer_map)
        return transfer_map

    def get_orbit(self, station_list):
        if station_list:
            orbit = [station_list[0][var] for var in self.vars]
        else:
            orbit = [0 for var in self.vars]
        return orbit

    def get_tune(self, tm):
        cosmu = (tm[0,0]+tm[1,1])/2.0
        if abs(cosmu) > 1.0:
            tune = cosmu
        else:
            angle = math.acos(cosmu)
            # if M*(1, 0) < 0 then one cell puts a particle in bottom two 
            # quadrants i.e. phase advance > pi 
            if tm[1,0] > 0:
                angle = 2*math.pi - angle
            tune = angle/2.0/math.pi
            #print("Get tune cosmu", cosmu, "phi", angle, "x1", tm[1,0], "nu", tune)
        return tune

    def get_score(self, hit_list_of_lists):
        source_hits, station_hits = self.get_station_hits(hit_list_of_lists)
        try:
            self.orbit = self.get_orbit(station_hits)
            self.tm = self.get_tm(source_hits, station_hits)
            for i in range(self.n_tunes):
                sub_matrix = numpy.array(self.tm)[2*i:2*i+2, 2*i:2*i+2]
                self.tunes[i] = self.get_tune(sub_matrix)
            if self.target_tune:
                self.score = [((self.tunes[i]-self.target_tune[i])/self.tune_tolerance[i])**2 for i in range(self.n_tunes)]
            else:
                self.score = [0 for i in range(self.n_tunes)]
        except Exception:
            sys.excepthook(*sys.exc_info())
            self.score = [10, 10]
        return self.score

    def save_score(self):
        output = {
            "score_type":"tune",
            "station":self.station,
            "orbit":self.orbit,
            "tunes":self.tunes,
            "target":self.target_tune,
            "transfer_matrix":self.tm.tolist(),
            "weighted_tunes":self.score
        }
        return output

    def print(self):
        print("Tune  ", end="")
        print(self.station, end=" | ")
        for tune in self.tunes:
            print(format(tune, self.fmt), end=" ")
        print(" | ", end="")
        for s in self.score:
            print(format(s, self.fmt), end=" ")
        print("|")

    @classmethod
    def setup(cls, config, optimisation):
        if "target_tune" not in optimisation: 
            return []
        score_list = []
        for item in sorted(optimisation["target_tune"].items()):
            score_list.append(TuneScore(item, config, optimisation))
        return score_list

class Parameter(object):
    def __init__(self):
        self.name = ""
        self.key = ""
        self.seed = 0
        self.lower_limit = 0
        self.upper_limit = 0
        self.error = 1e-3
        self.fixed = False
        self.minuit_index = -1
        self.current_value = -1
        self.current_error = -1

    def setup(self, parameter_dict):
        if self.current_error > 0:
            self.seed = self.current_value
            self.error = self.current_error
        for key, value in parameter_dict.items():
            if key not in self.__dict__:
                raise KeyError(f"Did not recognise parameter {key}")
            self.__dict__[key] = value

    def setup_minuit(self, index, minuit):
        minuit.DefineParameter(index, self.name,
                               self.seed, abs(self.error),
                               self.lower_limit, self.upper_limit)
        if self.fixed:
            minuit.FixParameter(index)
        self.minuit_index = index

    def update_from_minuit(self, minuit):
        value = ctypes.c_double()
        error = ctypes.c_double()
        minuit.GetParameter(self.minuit_index, value, error)
        self.current_value = value.value
        self.current_error = error.value

    def update_to_subs(self, subs):
        subs[self.key] = self.current_value

    def to_dict(self):
        return self.__dict__

class FindBumpParameters(object):
    def __init__(self, config):
        self.config = config
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.find_bump_parameters["run_dir"])
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

    def setup_minuit(self):
        global BUMP_FINDER
        par_dict = dict([(par.key, par) for par in self.parameters])
        for parameter_dict in self.optimisation["parameters"]:
            key = parameter_dict["key"]
            if key in par_dict:
                parameter = par_dict[key]
                parameter.setup(parameter_dict)
            else:
                parameter = Parameter()
                parameter.setup(parameter_dict)
                self.parameters.append(parameter)
        self.minuit = ROOT.TMinuit(len(self.optimisation["parameters"]))
        for i, parameter in enumerate(self.parameters):
            parameter.setup_minuit(i, self.minuit)
        BUMP_FINDER = self
        self.minuit.SetFCN(minuit_function)
        self.minuit_function()

    def store_data(self):
        outname = self.get_filename_root()+"_"+f'{self.store_index:03}'+".out"
        tmpname = self.get_filename_root()+".tmp"
        print("Moving data from", tmpname, "to", outname)
        os.rename(tmpname, outname)
        self.store_index += 1

    def find_bump_parameters(self):
        try:
            os.rename(
                self.get_filename_root()+".tmp",
                self.get_filename_root()+".old"
            )
        except OSError:
            pass
        for i, subs in enumerate(self.config.substitution_list):
            self.subs = subs
            for opt_i, optimisation in enumerate(self.config.find_bump_parameters["staged_optimisation"]):
                self.opt_i = {
                    "substitution_index":i,
                    "optimisation_stage":opt_i
                }
                self.optimisation = copy.deepcopy(optimisation)
                self.do_one_optimisation()

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

    def do_one_optimisation(self):
        print("Doing optimisation")
        self.overrides = self.config.find_bump_parameters["subs_overrides"]
        self.max_iterations = self.config.find_bump_parameters["max_iterations"]
        self.iteration = 0
        algorithm = self.config.find_bump_parameters["algorithm"]
        self.score_dict = Score.setup_scores(self.config, self.optimisation)
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
            self.track()
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
        target_n_hits = self.config.find_bump_parameters["target_n_hits"]
        n_hits = len(hit_list)
        penalty_factor = self.config.find_bump_parameters["penalty_factor"]
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
        var_list = self.config.find_bump_parameters["vars"]
        delta_dict = self.config.find_bump_parameters["deltas"]
        co = self.config.find_bump_parameters["closed_orbit"]
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
        ref_probes = self.config.find_bump_parameters["ref_probe_files"]
        ref_energy = self.config.find_bump_parameters["energy"]
        # fix momentum
        self.tracking = utilities.setup_tracking(self.config, ref_probes, ref_energy)
        hit_list = []
        for psv in psv_list:
            my_hit = copy.deepcopy(self.tracking.ref)
            for var in psv:
                my_hit[var] = psv[var]
            if "x'" in self.config.find_bump_parameters["vars"]:
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
    find_bump = FindBumpParameters(config)
    find_bump.find_bump_parameters()
    if __name__ == "__main__":
        input()
