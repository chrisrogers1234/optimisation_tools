"""
Generalised optimiser
"""

#import time
import os
#import sys
#import copy
#import json
#import math
import warnings
warnings.filterwarnings("ignore")

#import xboa.common
#from xboa.hit import Hit
#sys.path.insert(1, "scripts")
from optimisation_tools.analysis.optimiser import optimiser
from optimisation_tools.analysis.optimiser import parameter
from optimisation_tools.analysis.optimiser import score
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
        for an_optimiser in self.optimisation_stages:
            print("Running ", an_optimiser)
            an_optimiser.run_optimisation()

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

class TestConfig:
    run_control = {
        "output_dir":"output/test",
        "ffa_version":f"test rectilinear",
        "root_verbose":6000,
        "faint_text":'\033[38;5;243m',
        "default_text":'\033[0m',
        "random_seed":0,
    }
    substitution_list = [{}]
    optimisation = {
        "output_file":"find_bump_parameters",
        "staged_optimisation":[{ # get BF
                "parameters":[
                    {"name":"h bump 1", "key":"__h_bump_1_field__", "seed":0.0, "error":0.1, "fixed":False},
                    {"name":"h bump 2", "key":"__h_bump_2_field__", "seed":0.0, "error":0.1, "fixed":False},
                    {"name":"h bump 3", "key":"__h_bump_3_field__", "seed":0.0, "error":0.1, "fixed":True},
                    {"name":"h bump 4", "key":"__h_bump_4_field__", "seed":0.0, "error":0.1, "lower_limit":-1.0, "upper_limit":1.0, "fixed":True},
                    {"name":"h bump 5", "key":"__h_bump_5_field__", "seed":0.0, "error":0.1, "lower_limit":-1.0, "upper_limit":1.0, "fixed":True},
                ],
                "score":[{
                    "type":"psv",
                    "station":station,
                    "vector":[0, 0, 0, 0],
                    "tolerance":[1e-3, 1e-6, 1e-3, 1e-6],
                    "variables":["x", "px", "y", "py"],
                } for station in [0, 1, 2, 3]],
                "optimiser":{
                    "type":"minuit",
                    "max_iterations":1000,
                    "target_score":0.001,
                    "algorithm":"simplex",
                },
                "tracking":{
                    "subs":{
                        "__n_turns__":1.11,
                        "__do_magnet_field_maps__":False,
                        "__do_bump__":True,
                        "__do_foil__":False,
                        "__do_rf__":False,
                        "__n_particles__":3,
                    },
                    "probe_files":[],
                    "ref_energy":3.0,
                    "tmp_dir":"optimisation",
                    "mass_shell_condition":"p",
                }
            },
        ],
    }
    tracking = {
        "mpi_exe":None, #os.path.expandvars("${OPAL_ROOT_DIR}/external/install/bin/mpirun"),
        "beam_file_out":"beam.tmp",
        "n_cores":4,
        "links_folder":[], # link relative to lattice/VerticalFFA.in
        "lattice_file":[
            os.path.join(os.getcwd(), "lattice/github_demonstrator/parameter_overloads.in"),
            os.path.join(os.getcwd(), "cooling_demonstrator_github/cooling/g4bl/cooling.g4bl"),
            os.path.join(os.getcwd(), "cooling_demonstrator_github/cooling/g4bl/solenoid_field_map.txt"),
        ],
        "lattice_file_out":["cooling.g4bl", "cooling_lattice.g4bl", "solenoid_field_map.txt"],
        "tracking_code":"g4bl",
        "g4bl_path":os.path.expandvars("${G4BL_EXE_PATH}/g4bl"), # only used of opal path is None
        "g4bl_beam_format":"g4beamline_bl_track_file",
        "g4bl_output_format":"icool_for009",
        "opal_path":None,
        "tracking_log":"log",
        "flags":[],
        "step_size":1.,
        "ignore_events":[],
        "pdg_pid":-13,
        "clear_files":"*.h5",
        "verbose":10,
        "file_format":"hdf5",
        "analysis_coordinate_system":"none",
        "dt_tolerance":-1., # ns
        "station_dt_tolerance":-1., # ns, if +ve and two hits are close together, reallocate station
        "py_tracking":{
            "derivative_function":"u_derivative",
            "atol":1e-12,
            "rtol":1e-12,
            "verbose":True,
        }
    }


if __name__ == "__main__":
    main(TestConfig())
