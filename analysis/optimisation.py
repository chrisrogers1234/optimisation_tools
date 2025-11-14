"""
Generalised optimiser
"""

import os
import sys
#import warnings
#warnings.filterwarnings("ignore")
from optimisation_tools.analysis.optimiser import optimiser
from optimisation_tools.analysis.optimiser import parameter
from optimisation_tools.analysis.optimiser import score
from optimisation_tools.analysis.optimiser import bunch_score
from optimisation_tools.analysis.optimiser import tracking_wrapper

class Optimisation:
    def __init__(self):
        self.output_file = None
        self.energy = 0.0
        self.parameters = []
        self.tracking = None
        self.final_subs_overrides = {}

    def setup(self, optimisation_config):
        """Set global configuration parameters"""
        for key in optimisation_config.keys():
            if key not in self.config_exclusions and key not in self.__dict__:
                raise KeyError(f"Did not recognise {key} while setting up score, should be in {self.__dict__.keys()}")
            self.__dict__[key] = optimisation_config[key]

    def setup_optimiser(self, config):
        self.setup(config.optimisation)
        self.optimisation_stages = []
        for optimisation_stage in config.optimisation["staged_optimisation"]:
            parameter_list = []
            for parameter_config in optimisation_stage["parameters"]:
                parameter_list.append(parameter.Parameter.setup(parameter_config))
            an_optimiser = optimiser.Optimiser.setup_optimisations(optimisation_stage["optimiser"])
            an_optimiser.setup(optimisation_stage["optimiser"])
            an_optimiser.parameter_list = parameter_list
            an_optimiser.score = score.Score.setup_scores(optimisation_stage["score"])
            an_optimiser.tracking = tracking_wrapper.TrackingWrapper()
            an_optimiser.tracking.setup(config, optimisation_stage["tracking"])
            self.optimisation_stages.append(an_optimiser)

    def optimise(self):
        post_dict = {}
        for an_optimiser in self.optimisation_stages:
            print(f"\n\n=================== Running '{an_optimiser.name}'  ===================")
            try:
                # accumulate post_dict over all the stages
                post_dict.update(an_optimiser.post_output())
                an_optimiser.receive_output(post_dict)
                # we post and receive twice in case there are some messages
                # that need to be received before the next one can be posted
                post_dict.update(an_optimiser.post_output())
                an_optimiser.receive_output(post_dict)
                # run the actual optimisation
                an_optimiser.run_optimisation()
                # update following optimisation
                post_dict.update(an_optimiser.post_output())
            except Exception:
                sys.excepthook(*sys.exc_info())


    def store_data(self):
        outname = self.get_filename_root()+"_"+f'{self.store_index:03}'+".out"
        tmpname = self.get_filename_root()+".tmp"
        print("Moving data from", tmpname, "to", outname)
        os.rename(tmpname, outname)
        self.store_index += 1

    config_exclusions = ["staged_optimisation"]

global BUMP_FINDER
def minuit_function(nvar, parameters, score, jacobian, err):
    global BUMP_FINDER
    BUMP_FINDER.minuit_function(nvar, parameters, score, jacobian, err)


def main(config):
    optimisation = Optimisation()
    optimisation.setup_optimiser(config)
    optimisation.optimise()

co = [0.0, 0.0, 0.0, 0.0]
bump = [0.0, 1.0, 0.0, 0.0]
bz = 0.2

if __name__ == "__main__":
    main(TestConfig())
