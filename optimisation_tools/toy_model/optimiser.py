import sys
import os
import math
import time
import json

import matplotlib
import numpy
import platypus
from platypus import Problem
from optimisation_tools.toy_model._toy_model import ToyModel

class ToyModelOptimisation(object):
    def __init__(self, toy_config):
        self.model = ToyModel()
        self.model.do_plots = False
        self.model.verbose = -1
        self.model.parse_args()
        if toy_config != None:
            self.model.do_config(toy_config)
        # angle_1 angle_2 foil_angle beta alpha beta alpha
        self.is_fixed = [  False,      False,    True,     False, False, False, False]
        self.seed = [0.0, -math.pi/2., math.pi,       2.0,   0.0,   2.0,   0.0]
        self.min = [    -math.pi,   -math.pi, math.pi*0.5, 0.5,  -5.0,   0.5,  -5.0]
        self.max = [    +math.pi,    math.pi, math.pi*1.5, 5.0,   5.0,   5.0,   5.0]
        self.sigma = 1.
        self.n_trials = None
        self.k_index = 1.0
        self.final_event_multiplier = 1 # final run gets n times more per pulse
        self.iteration = 0
        self.algorithm = None
        self.iterations = []
        self.target_emittance = 8.0
        self.output_dir = self.model.output_dir
        self.toy_config = toy_config
        self.opt_config = None
        self.bumps_function = None
        self.foil_angle = None # WARNING DUMMY VARIABLE

    def do_config(self, optimiser_config):
        for key in optimiser_config:
            if key in self.__dict__ or key:
                self.__dict__[key] = optimiser_config[key]
            else:
                raise(KeyError("Did not recognise config key '"+str(key)+"'"))
        self.opt_config = optimiser_config

    def optimise(self, config_list_of_dicts):
        self.algorithm = self.setup_optimisation_cmaes()
        self.iteration = 0
        seed = [x for i, x in enumerate(self.seed) if not self.is_fixed[i]]
        self.run_one(seed)
        self.model.print_output()
        if self.n_trials > 0:
            try:
                self.algorithm.run(self.n_trials)
            except Exception:
                sys.excepthook(*sys.exc_info())
        self.iterations = sorted(self.iterations, key = lambda x: x[0][0])
        best = self.iterations[0]
        print("Running best inputs", best[1], "with score", best[0])
        self.model.do_plots = True
        self.model.number_per_pulse *= self.final_event_multiplier
        seed = [x for i, x in enumerate(best[1]) if not self.is_fixed[i]]
        self.iteration = 0
        self.run_one(seed, {"plot_frequency":1, "do_movie":True})
        self.finalise(config_list_of_dicts)
        #matplotlib.pyplot.show(block=False)

    def finalise(self, config_list_of_dicts):
        fout = open(self.output_dir+"/optimiser.json", "w")
        out = {
            "toy_config":self.toy_config,
            "opt_config":self.opt_config,
            "iterations":self.iterations
        }
        print(json.dumps(out), file=fout)

        params = self.model.get_output_parameters(config_list_of_dicts)
        for key, value in params.items():
            try:
                print(key, format(value, "6.4g"))
            except Exception:
                print(key, type(value))
        print()
        fout = open(self.output_dir+"/run_summary.json", "w")
        fout.write(json.dumps(params))

    def straight_bumps(self, angle_1, angle_2, foil_angle, target_position, n_pulses, n_trajectory_turns):
        default_bumps = [["coupled", 0.0, 0.0, 0.0, 0.0]]
        for i in range(n_trajectory_turns-1):
            dr = target_position*(i+1)/(n_trajectory_turns-1)
            default_bumps.append(["coupled",
                              -dr*math.sin(math.radians(foil_angle)), 0.0,
                              -dr*math.cos(math.radians(foil_angle)), 0.0])
        for i in range(self.n_tracking_turns):
            default_bumps.append(default_bumps[-1])
        self.model.default_bumps = default_bumps
        my_range = [i for i, bump in enumerate(default_bumps)]
        self.model.turn_bumper_index = [my_range, my_range]

    def correlated_default_bumps(self, angle_1, angle_2, foil_angle, target_emittance, n_pulses, n_trajectory_turns):
        if n_pulses == 1:
            default_bumps = [["action_angle",
                              angle_1, 0.0,
                              angle_2, 0.0] for i in range(n_pulses)]
        else:
            default_bumps = [["action_angle",
                              angle_1, i**self.k_index*target_emittance/(n_pulses-1)**self.k_index,
                              angle_2, i**self.k_index*target_emittance/(n_pulses-1)**self.k_index]
                                         for i in range(n_pulses)]
        delta = 30.0 # mm; check sign of sin
        foil_angle = math.radians(foil_angle)
        traj_end = ["", delta*math.sin(foil_angle), 0.0, delta*math.cos(foil_angle), 0.0]
        # ["action_angle"]+\
        for j in range(1, n_trajectory_turns+1):
            default_bumps.append(["delta"]+[traj_end[i]/n_trajectory_turns for i in range(1, 5)])
        self.model.default_bumps = default_bumps
        my_range = [i for i, bump in enumerate(default_bumps)]
        self.model.turn_bumper_index = [my_range, my_range]

    def anticorrelated_default_bumps(self, angle_1, angle_2, foil_angle, target_emittance, n_pulses, n_trajectory_turns):
        #n_pulses is the number of turns over which beam is inserted
        #n_trajectory_turns is the number of turns over which we move the beam
        #self.n_tracking_turns is the number of turns over which we just watch the beam propagate
        if n_pulses == 1:
            default_bumps = [["action_angle",
                              angle_1, target_emittance/2.0,
                              angle_2, target_emittance/2.0]]
        else:
            default_bumps = [["action_angle", 
                          angle_1, i*target_emittance/(n_pulses-1),
                          angle_2, (n_pulses-1-i)*target_emittance/(n_pulses-1)]
                                     for i in range(n_pulses)]
        delta = 20.0 # mm; check sign of sin
        foil_angle = math.radians(foil_angle)
        traj_end = ["", delta*math.sin(foil_angle), 0.0, delta*math.cos(foil_angle), 0.0]
        for j in range(1, n_trajectory_turns+1):
            default_bumps.append(["delta"]+[traj_end[i]/n_trajectory_turns for i in range(1, 5)])
        self.model.default_bumps = default_bumps
        my_range = [i for i, bump in enumerate(default_bumps)]
        self.model.turn_bumper_index = [my_range, my_range]

    def add_fixed(self, arguments):
        my_args = []
        index = 0
        for i, is_fixed in enumerate(self.is_fixed):
            if not is_fixed:
                my_args.append(arguments[index])
                index += 1
            else:
                my_args.append(self.seed[i])
        return my_args

    def run_one(self, arguments, overrides=None):
        arguments = self.add_fixed(arguments)
        self.iteration += 1
        if self.n_trials and self.iteration > self.n_trials:
            raise RuntimeError("Out of iterations "+str(self.iteration)+"/"+str(self.n_trials))
        print("Run", self.iteration, "with args", [format(arg, "8.4g") for arg in arguments], end='')
        n_pulses = self.model.number_pulses
        n_trajectory_turns = self.model.max_turn-self.n_tracking_turns
        angle_1 = arguments[0]
        angle_2 = arguments[1]
        foil_angle = math.degrees(arguments[2])
        beta_x, alpha_x = arguments[3], arguments[4]
        beta_y, alpha_y = arguments[5], arguments[6]
        self.bumps_function(angle_1, angle_2, foil_angle, self.target_emittance, n_pulses, n_trajectory_turns)
        self.model.foil_angle = foil_angle   
        self.model.beta_x = abs(beta_x)
        self.model.alpha_x = alpha_x
        self.model.beta_y = abs(beta_y)
        self.model.alpha_y = alpha_y
        self.model.initialise()
        if overrides:
            self.model.do_config(overrides)
        self.model.run_toy_model()
        self.model.finalise()
        mean_foil_hits = self.model.output["mean_foil_hits"]
        n_outside_acceptance = self.model.output["n_outside_acceptance"]*1.0/self.model.output["n_events"]
        print("    ... Mean foil hits", format(mean_foil_hits, "6.4g"), "n outside", format(n_outside_acceptance, "8.4g"))
        score = [n_outside_acceptance+mean_foil_hits/10.0] 
        self.iterations.append([score, arguments])
        return score

    def setup_optimisation_cmaes(self):
        n_parameters = len([is_fixed for is_fixed in self.is_fixed if not is_fixed])
        n_variables = 1
        problem = platypus.Problem(n_parameters, n_variables)
        min_ = [x for i, x in enumerate(self.min) if not self.is_fixed[i]]
        max_ = [x for i, x in enumerate(self.max) if not self.is_fixed[i]]
        seed_ = [x for i, x in enumerate(self.seed) if not self.is_fixed[i]]
        problem.types = [platypus.Real(min_[i], max_[i]) for i in range(n_parameters)]
        problem.directions[:] = Problem.MINIMIZE
        problem.function = self.run_one
        algorithm = platypus.CMAES(problem)
        algorithm.initial_search_point = seed_
        algorithm.sigma = self.sigma # note this is a single value
        return algorithm

    n_tracking_turns = 50


def do_toy_config(dp_over_p, col_density, emit, n_pulses, n_trajectory_turns, study, version):
    print("Adding toy config with dp/p", dp_over_p, "foil", col_density, 
          "emittance", emit, "n_pulses", n_pulses,
          "n_traj", n_trajectory_turns)
    n_turns = n_trajectory_turns+ToyModelOptimisation.n_tracking_turns
    my_range = [i for i in range(n_turns)]
    my_config = {
            "lattice":"2022-02-01 hffa",
            "study":study,
            "version":version,
            "dp_over_p":dp_over_p,
            "foil_column_density":col_density,
            "pulse_emittance":emit,
            "number_pulses":n_pulses,
            "max_turn":n_turns,
            "turn_bumper_index":[my_range, my_range],
            "number_per_pulse":round(1000/n_pulses),
            "plot_frequency":n_turns,
            "stats_frequency":n_turns,
            "injection_ellipse_algorithm":"from_twiss",
            "closed_orbit":"output/baseline/closed_orbits_cache",
            "verbose_particles":[],
            "rf_voltage":0.006,
        }
    return my_config

 
def do_opt_config(k_index, target_emit, n_trials, final_event_multiplier, is_fixed, seed = None):
    print("Adding optimiser config with k ", k_index, "target_emit", target_emit,
          "n_trials", n_trials, "final multiplier", final_event_multiplier)
    my_dict = {
        "k_index":k_index,
        "target_emittance":target_emit,
        "n_trials":n_trials,
        "final_event_multiplier":final_event_multiplier,
        "is_fixed":is_fixed,
        "foil_angle":round(seed[2], 3),
    }
    if seed != None:
        my_dict["seed"] = seed
    return my_dict

def dir_name(opt_config, toy_config, name_base, foil_angle):
    dir_keys = {
        "dp_over_p":"dp_p",
        "foil_column_density":"col_dens",
        "pulse_emittance":"inj_emit",
        "number_pulses":"n_i",
        "max_turn":"n_tot",
        "k_index":"k",
        "target_emittance":"tgt_emit",
        "foil_angle":"foil",
    }
    name = ""
    for config in [opt_config, toy_config]:
        for key, value in sorted(dir_keys.items()):
            if key in config:
                name += value+"_"+str(config[key])+"__"
    name = name_base+"/"+name[:-2]+"/"
    opt_config["output_dir"] = name
    toy_config["output_dir"] = name
    print("Running in dir", name)


def correlated_painting():
    config_list = []
    for n in [30]:#, 10, 20, 30, 50]:
        for emit in [0.25e-3]:#, 0.026e-3]:
            for thickness in [20e-6]: # <----------
                for foil_angle in [math.pi/2]:#, math.pi]:
                    for target_emit in [8.0*1e-3]:
                        u_angle = -math.pi/2.0
                        v_angle = 0.0
                        seed = [u_angle, v_angle, foil_angle, 2.0, 0.0, 2.0, 0.0]
                        is_fixed = [False]*2+[True]*1+[False]*4
                        toy_config = do_toy_config(0.0013, thickness, emit, n, n, "correlated painting", "v1")
                        opt_config = do_opt_config(1.0, target_emit, 1000, 10, is_fixed, seed)
                        name_base = "output/2022-03-01_baseline/correlated_painting/toy_model_3"
                        dir_name(toy_config, opt_config, name_base, foil_angle)
                        config_list.append((opt_config, toy_config))
    time.sleep(10)
    for optimiser_config, toy_config in config_list:
        print("Doing correlated painting")
        optimisation = ToyModelOptimisation(toy_config)
        optimisation.do_config(optimiser_config)
        optimisation.bumps_function = optimisation.correlated_default_bumps
        optimisation.optimise([optimiser_config, toy_config])


def anticorrelated_painting():
    config_list = []
    for n in [25, 10, 20, 30, 40, 50]:
        for emit in [0.026e-3, 0.25e-3]:
            for thickness in [20e-6]:
                for target_emit in [8.0*1e-3]:
                    for foil_angle in [math.pi/2, math.pi]:
                        seed = [0.0, 0.0, foil_angle, 2.0, 0.0, 2.0, 0.0]
                        is_fixed = [False]*2+[True]*1+[False]*4
                        toy_config = do_toy_config(0.0013, thickness, emit, n, n, "anticorrelated painting", "v1")
                        opt_config = do_opt_config(1.0, target_emit, 1000, 10, is_fixed, seed)
                        name_base = "output/2022-03-01_baseline/anticorrelated_painting/toy_model_2/"
                        dir_name(toy_config, opt_config, name_base, foil_angle)
                        config_list.append((opt_config, toy_config))
    time.sleep(10)
    for optimiser_config, toy_config in config_list:
        print("Doing anticorrelated painting")
        optimisation = ToyModelOptimisation(toy_config)
        optimisation.do_config(optimiser_config)
        optimisation.bumps_function = optimisation.anticorrelated_default_bumps
        optimisation.optimise([optimiser_config, toy_config])



def single_turn_injection():
    config_list = []
    n_iterations = 1000
    for emit in [0.026e-3, 0.25e-3]:
        for n in [25, 10, 20, 30, 40, 50]:
            for foil_angle in [math.pi/2, math.pi]:
                for thickness in [20e-6]:#, 5e-6]:
                    #angle_1, angle_2, foil_angle, beta_x, alpha_x, beta_y, alpha_y
                    seed = [0.0, 0.0, foil_angle, 2.0, 0.0, 2.0, 0.0]
                    is_fixed = [True]*3+[False]*4
                    toy_config = do_toy_config(0.0013, thickness, emit, 1, n, "single turn injection", "v1") # inject in 1 pulse and then take n turns to move the trajectory
                    opt_config = do_opt_config(1.0, -25.0, n_iterations, 10, is_fixed, seed) #k_index, displacement, n optimiser steps, final_event_multiplier, is_fixed, seed
                    name_base = "output/2022-03-01_baseline/single_turn_injection/toy_model_2/"
                    dir_name(toy_config, opt_config, name_base, foil_angle)
                    config_list.append((opt_config, toy_config))
    for optimiser_config, toy_config in config_list:
        optimisation = ToyModelOptimisation(toy_config)
        optimisation.do_config(optimiser_config)
        optimisation.bumps_function = optimisation.straight_bumps
        optimisation.optimise([optimiser_config, toy_config])

def load_and_redo():
    bumps_function = "correlated"
    input_dir = "output/arctan_baseline/correlated_painting/toy_model/dp_p_0.0013__col_dens_2e-05__n_tot_100__n_i_50__inj_emit_0.026__foil_1.571__k_1.0__tgt_emit_8.0/"
    new_output_dir = "output/arctan_baseline/correlated_painting/toy_model/redo/dp_p_0.0013__col_dens_2e-05__n_tot_100__n_i_50__inj_emit_0.026__foil_1.571__k_1.0__tgt_emit_8.0/"
    fin = open(os.path.join(input_dir, "optimiser.json"))
    optimiser_in = json.loads(fin.read())
    optimiser_config = optimiser_in["opt_config"]
    toy_config = optimiser_in["toy_config"]
    # hack the output directory so we don't overwrite anything
    optimiser_config["output_dir"] = new_output_dir
    toy_config["output_dir"] = new_output_dir
    # fix the seed to previous optimisation and n_trials to 0
    optimiser_config["seed"] = optimiser_in["iterations"][0][1]
    optimiser_config["n_trials"] = 0
    # increase plot verbosity
    toy_config["plot_frequency"] = 1
    optimisation = ToyModelOptimisation(toy_config)
    optimisation.do_config(optimiser_config)
    if bumps_function == "anticorrelated":
        optimisation.bumps_function = optimisation.anticorrelated_default_bumps
    elif bumps_function == "correlated":
        optimisation.bumps_function = optimisation.correlated_default_bumps
    optimisation.optimise([optimiser_config, toy_config])


def main():
    #load_and_redo()
    #single_turn_injection()
    correlated_painting()
    #anticorrelated_painting()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")

if __name__ == "__main__":
    main()
