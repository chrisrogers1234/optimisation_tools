import numbers
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

    def post_output(self):
        return {}

    def receive_output(self, parameter_list):
        pass

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
        self.name = "No name"
        self.type = "minuit"
        self.algorithm = "Migrad"
        self.target_score = 1e9
        self.max_iterations = 1000
        self.iteration = 0
        self.parameter_list = []
        self.tracking = None
        self.score = None
        self.minuit = None
        self.iteration_number = 0
        self.message_dict = {}

    def setup(self, optimiser_config):
        self.minuit = ROOT.TMinuit()
        self.minuit.SetFCN(minuit_function)
        for key in optimiser_config:
            if key not in self.__dict__:
                raise KeyError("did not recognise", key)
            self.__dict__[key] = optimiser_config[key]

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
        self.iteration_number = 0
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
        except StopIteration:
            print("Reached maximum iteration", self.max_iterations, "without convergence\n\n")
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

    def update_parameters_from_minuit(self):
        for i, parameter in enumerate(self.parameter_list):
            parameter.update_from_minuit(i)

    def get_score(self, hit_list_of_lists):
        total_score = self.score.get_score()
        return total_score

    def minuit_function(self, nvar=None, parameters=None, score=ctypes.c_float(0.0), jacobian=None, err=None):
        self.iteration_number += 1
        if self.iteration_number > self.max_iterations:
            raise StopIteration("Run out of iterations")
        for i, parameter in enumerate(self.parameter_list):
            self.update_parameter(i)
        post_dict = self.post_output()
        self.receive_output(post_dict)
        hit_list_of_lists = self.tracking.track(self.parameter_list, self.score)
        a_score = self.score.get_score(hit_list_of_lists)
        score.value =  a_score
        print(f"Total score: {a_score:10.4g}  Iteration {self.iteration_number}/{self.max_iterations}\n")

    def receive_output(self, post_dict):
        """
        Messaging protocol for messaging between parameters and scores and over
        different optimisation stages
        """
        print("Reconciling parameters against", len(post_dict), "found parameters")
        self.message_dict.update(post_dict)
        for parameter in self.parameter_list:
            parameter.receive_output(post_dict)
        print(post_dict)
        self.tracking.receive_output(post_dict)
        self.score.receive_output(post_dict)

    def post_output(self):
        for parameter in self.parameter_list:
            self.message_dict.update(parameter.post_output())
        self.message_dict.update(self.score.post_output())
        return self.message_dict

    root_algorithms = ["simplex", "migrad"]
    minuit_global = None