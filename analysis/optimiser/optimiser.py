import sys
import time
import ctypes
import copy

import ROOT

from optimisation_tools.opal_tracking import OpalTracking

class Optimiser:
    def __init__(self):
        pass

    def setup(self, optimisation_config):
        pass

    def run_optimisation(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def setup_optimisations(cls, optimiser_config):
        optimiser_types = {
            "minuit":MinuitOptimiser,
        }
        a_type = optimiser_config["type"]
        optimiser = optimiser_types[a_type]()
        optimiser.setup(optimiser_config)
        return optimiser

def minuit_function(*args):
    try:
        MinuitOptimiser.minuit_global.minuit_function(*args)
    except Exception:
        print("Score function failed")
        sys.excepthook(*sys.exc_info())
        raise


class MinuitOptimiser(Optimiser):
    def __init__(self):
        super(Optimiser).__init__()
        self.algorithm = "Migrad"
        self.target_score = 1e9
        self.max_iterations = 1000
        self.iteration = 0
        self.parameter_list = []
        self.tracking = None
        self.score = None
        self.minuit = None

    def setup(self, optimiser_config):
        self.minuit = ROOT.TMinuit()
        self.minuit.SetFCN(minuit_function)

    def setup_parameter(self, index):
        parameter = self.parameter_list[index]
        self.minuit.DefineParameter(index, parameter.name,
                               parameter.seed, abs(parameter.error),
                               parameter.lower_limit, parameter.upper_limit)
        if parameter.fixed:
            self.minuit.FixParameter(index)
        parameter.index = index

    def prerun_setup(self):
        for i, parameter in enumerate(self.parameter_list):
            self.setup_parameter(i)
        MinuitOptimiser.minuit_global = self

    def update_parameter(self, index):
        value = ctypes.c_double()
        error = ctypes.c_double()
        self.minuit.GetParameter(index, value, error)
        self.parameter_list[index].current_value = value.value
        self.parameter_list[index].current_error = error.value

    def run_optimisation(self):
        print("setup minuit")
        self.prerun_setup()
        print("run minuit")
        try:
            self.minuit.Command(self.algorithm+" "+str(self.max_iterations)+" "+str(self.target_score))
        except Exception:
            sys.excepthook(*sys.exc_info())
            print("Minuit failed")
            raise
        print("done minuit")

    def teardown(self):
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
        for i, parameter in enumerate(self.parameter_list):
            parameter.update_from_minuit(i)

    def get_score(self, hit_list_of_lists):
        total_score = self.score.get_score()
        return total_score

    def minuit_function(self, nvar=None, parameters=None, score=ctypes.c_float(0.0), jacobian=None, err=None):
        for i, parameter in enumerate(self.parameter_list):
            self.update_parameter(i)
        hit_list_of_lists = self.tracking.track(self.parameter_list, self.score)
        a_score = self.score.get_score(hit_list_of_lists)
        score.value =  a_score
        print("Total score:", a_score, "\n")

    root_algorithms = ["simplex", "migrad"]
    minuit_global = None