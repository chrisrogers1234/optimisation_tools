"""
Generalised optimiser
"""

import json
import os
import sys
from optimisation_tools.analysis.optimiser import optimiser_base
from optimisation_tools.analysis.optimiser import differential_evolution
from optimisation_tools.analysis.optimiser import minuit
from optimisation_tools.analysis.optimiser import minimization
from optimisation_tools.analysis.optimiser import parameter
from optimisation_tools.analysis.optimiser import score
from optimisation_tools.analysis.optimiser import bunch_score
from optimisation_tools.analysis.optimiser import tracking_wrapper
from optimisation_tools.utils import utilities

class Optimisation:
    def __init__(self):
        self.output_file = None
        self.allow_restart = 0
        self.optimisation_stages = []
        self.config = None

    def setup(self, optimisation_config):
        """Set global configuration parameters"""
        for key in optimisation_config.keys():
            if key not in self.config_exclusions and key not in self.__dict__:
                raise KeyError(f"Did not recognise {key} while setting up score, should be in {self.__dict__.keys()}")
            self.__dict__[key] = optimisation_config[key]

    def setup_optimiser(self, config):
        self.config = config
        self.setup(config.optimisation)
        self.optimisation_stages = []
        for optimisation_stage in config.optimisation["staged_optimisation"]:
            an_optimiser = self.optimiser_factory(optimisation_stage["optimiser"])
            an_optimiser.setup(optimisation_stage["optimiser"])
            try:
                parameter_list = []
                for parameter_config in optimisation_stage["parameters"]:
                    parameter_list.append(parameter.Parameter.setup(parameter_config))
                an_optimiser.config = config
                an_optimiser.parameter_list = parameter_list
                an_optimiser.score = score.Score.setup_scores(optimisation_stage["score"])
                an_optimiser.tracking = tracking_wrapper.TrackingWrapper()
                an_optimiser.tracking.setup(config, optimisation_stage["tracking"])
                self.optimisation_stages.append(an_optimiser)
            except Exception:
                print(f"Exception in {an_optimiser.name}")
                raise

    def optimiser_factory(cls, optimiser_config):
        optimiser_types = {
            "minuit":minuit.Minuit,
            "differential_evolution":differential_evolution.DifferentialEvolution,
            "scipy_minimisation":minimization.Minimization,
        }
        a_type = optimiser_config["type"]
        optimiser = optimiser_types[a_type]()
        optimiser.setup(optimiser_config)
        return optimiser


    def get_filename(self):
        fname = self.config.optimisation["output_file"]
        fname = f"{self.config.run_control['output_dir']}/{fname}"
        return fname

    def save_data(self, post_dict):
        fname = self.get_filename()
        post_dict = utilities.sanitize(post_dict)
        with open(fname, "w") as fout:
            json.dump(post_dict, fout, indent=2)
        print(f"Updated optimisation in {fname}")

    def load_data(self):
        fname = self.get_filename()
        with open(fname) as fin:
            try:
                post_dict = json.load(fin)
            except Exception:
                raise IOError("Failed to load json file")
        return post_dict


    def optimise(self):
        try:
            post_dict = self.load_data()
            print(json.dumps(post_dict, indent=2))
        except IOError:
            sys.excepthook(*sys.exc_info())
            post_dict = {"optimisation":{}}
        for an_optimiser in self.optimisation_stages:
            if an_optimiser.disable:
               print(f"\n\n=================== Skipping '{an_optimiser.name}'  ===================")
               print("     disable flag is true in config")
            elif self.allow_restart and an_optimiser.allow_restart and an_optimiser.name in post_dict["optimisation"]:
               print(f"\n\n=================== Skipping '{an_optimiser.name}'  ===================")
               print("     Found in optimisation file and allow_restart is true")
            else:
                print(f"\n\n=================== Running - '{an_optimiser.name}'  ====================")
                try:
                    print("Following optimisers found before", [key for key in post_dict["optimisation"]])
                    post_dict = self.run_optimisation(an_optimiser, post_dict)
                    print("Following optimisers found after ", [key for key in post_dict["optimisation"]])
                except Exception:
                    print("Exception with post_dict")
                    print(json.dumps(post_dict, indent=2))
                    #sys.excepthook(*sys.exc_info())
                    raise
            if not an_optimiser.accept_optimisation():
                print(f"'{an_optimiser.name}' failed; halting")
                break
            else:
                print(f"'{an_optimiser.name}' okay; continuing")

    def run_optimisation(self, an_optimiser, post_dict):
        # accumulate post_dict over all the stages
        post_dict = utilities.update(post_dict, an_optimiser.post_output())
        an_optimiser.receive_output(post_dict)
        # we post and receive twice in case there are some messages
        # that need to be received before the next one can be posted
        post_dict = utilities.update(post_dict, an_optimiser.post_output())
        an_optimiser.receive_output(post_dict)
        # run the actual optimisation
        an_optimiser.run_optimisation()
        # update following optimisation
        post_dict = utilities.update(post_dict, an_optimiser.post_output())
        # write to disk
        self.save_data(post_dict)
        print("Saved data for optimisations", [key for key in post_dict["optimisation"]])
        return post_dict

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
