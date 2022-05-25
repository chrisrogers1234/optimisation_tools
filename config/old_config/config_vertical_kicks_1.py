import math
import copy
import os
import json
from . import config_triplet_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        sub = config.get_baseline_substitution()
        sub["__h_bump_1_phi__"] = 1.00
        sub["__h_bump_2_phi__"] = 2.00
        sub["__h_bump_3_phi__"] = 3.07
        sub["__h_bump_4_phi__"] = 4.00
        sub["__h_bump_5_phi__"] = 5.00
        sub["__v_bump_1_phi__"] = 1.00
        sub["__v_bump_2_phi__"] = 2.00
        sub["__v_bump_3_phi__"] = 3.07
        sub["__v_bump_4_phi__"] = 4.00
        sub["__v_bump_5_phi__"] = 5.00
        sub["__h_bump_1_dphi__"] = 0.0
        sub["__h_bump_2_dphi__"] = 0.0
        sub["__h_bump_3_dphi__"] = 0.25
        sub["__h_bump_4_dphi__"] = 0.0
        sub["__h_bump_5_dphi__"] = 0.0
        sub["__v_bump_1_dphi__"] = 0.0
        sub["__v_bump_2_dphi__"] = 0.0
        sub["__v_bump_3_dphi__"] = 0.25
        sub["__v_bump_4_dphi__"] = 0.0
        sub["__v_bump_5_dphi__"] = 0.0
        sub["__foil_probe_phi__"] = 3.095
        sub["__foil_probe_dphi__"] = 0.25/2.+0.025*math.pi/10.

        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/triplet_baseline/vertical_kicks_2")
        self.run_control["find_closed_orbits_4d"] = True
        self.run_control["find_da"] = False
        self.run_control["find_bump_parameters"] = True
        self.run_control["track_bump"] = True

        self.tracking["dt_tolerance"] = 1.
        self.tracking["verbose"] = 0
        self.find_closed_orbits["final_subs_overrides"]["__do_bump__"] = True
        self.find_closed_orbits["final_subs_overrides"]["__cartesian_x_min__"] = -5.0
        self.find_closed_orbits["final_subs_overrides"]["__cartesian_dx__"] = 0.01

        x_tol, p_tol = 0.01, 0.001
        foil_orbit_no_hminus_kick = [3744.08063, 3.843,-89.81, 1.151] # orbit at the foil, if dipole is not included
        foil_orbit_hminus_0p10_kick = [3744.07493, 3.849,-92.17, 4.105] # orbit at the foil, after 0.10 T dipole
        foil_orbit_hminus_0p05_kick = [3744.06827, 3.842,-91.01, 2.63]  # orbit at the foil, after 0.05 T dipole
        foil_orbit = foil_orbit_hminus_0p10_kick
        co = [3739.427804804206, -0.0002874136417290174, -88.77890374233482, -0.0008126002995965109]
        target_bump = [ 4.02587004e+00, -9.31306890e-05,  7.95482423e+00,  8.68548964e-03]
        bump = [(0, [x*i/10. for x in target_bump]) for i in range(1)]
        self.find_bump_parameters["seed_errors"] = [1e-2]*10
        self.find_bump_parameters["closed_orbit"] = co
        self.find_bump_parameters["bump"] = bump
        self.find_bump_parameters["max_iterations"] = 1000
        self.find_bump_parameters["target_n_hits"] = 2
        self.find_bump_parameters["subs_overrides"] = {
            "__n_turns__":0.9,
            "__step_size__":0.001,
            "__do_magnet_field_maps__":False,
            "__do_bump__":True,
        }
        self.find_bump_parameters["algorithm"] = "simplex"
        self.find_bump_parameters["field_names"] = [
            "__v_bump_3_field__", "__h_bump_3_field__",
            "__v_bump_4_field__", "__h_bump_4_field__",
            "__v_bump_5_field__", "__h_bump_5_field__",
        ]
        self.find_bump_parameters["staged_optimisation"]= [{ # recover closed orbit
            "seed_fields":{
                        "__h_bump_3_field__":0.0,
                        "__v_bump_3_field__":0.1,
                        "__h_bump_4_field__":0.0,
                        "__v_bump_4_field__":0.0,
                        "__h_bump_5_field__":0.0,
                        "__v_bump_5_field__":0.0,
            },
            "target_orbit":dict([ #maps station:target orbit
                ]+[(i, co) for i in (1, 7, 8)]),
            "position_tolerance":x_tol,
            "momentum_tolerance":p_tol,
            "fix_bumps":["__v_bump_3_field__", "__h_bump_3_field__",
                         "__v_bump_-4_field__", "__h_bump_-4_field__",
                         "__v_bump_-5_field__", "__h_bump_-5_field__",],
        }]

        injection_orbit = [foil_orbit[i] for i in range(4)]
        index = 1
        self.track_bump["injection_orbit"] = [3744.0726767615297, 1.6650418416417807, -49.829990504267954, -6.205532738677988]
        self.track_bump["foil_phi"] = 110.8/36.0
        self.track_bump["foil_optimisation_stage"] = 0
        self.track_bump["field_optimisation_stage"] = 0
        self.track_bump["proton_orbit_station"] = 1
        self.track_bump["proton_orbit_phi"]= 0.
        self.track_bump["input_file"] = "find_bump_parameters_00?.out"
        self.substitution_list = []
        for v_bump_2 in [0.05*i for i in range(0, 5)]:
            my_sub = copy.deepcopy(sub)
            my_sub["__do_bump__"] = True
            my_sub["__v_bump_2_field__"] = v_bump_2
            self.substitution_list.append(my_sub)

        #self.find_closed_orbits["seed"] = [co]