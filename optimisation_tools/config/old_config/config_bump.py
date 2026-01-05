import math
import os
from . import config_sector_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        sub = config.get_baseline_substitution()
        sub["__h_bump_1_phi__"] = 0.07
        sub["__h_bump_2_phi__"] = 1.07
        sub["__h_bump_3_phi__"] = 2.07
        sub["__h_bump_4_phi__"] = 2.93
        sub["__h_bump_5_phi__"] = 3.93
        sub["__v_bump_1_phi__"] = 0.07
        sub["__v_bump_2_phi__"] = 1.07
        sub["__v_bump_3_phi__"] = 2.07
        sub["__v_bump_4_phi__"] = 2.93
        sub["__v_bump_5_phi__"] = 3.93
        self.substitution_list = [sub]


        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/sector_baseline/bumps_7")
        self.run_control["find_closed_orbits_4d"] = True
        self.run_control["find_da"] = False
        self.run_control["find_bump_parameters"] = True
        self.run_control["track_bump"] = True

        self.tracking["dt_tolerance"] = 10.
        self.tracking["verbose"] = 0

        x_tol, p_tol = 0.01, 0.001
        foil_orbit = [3897.85819, -17.77, 89.17, 3.897]
        co = [3967.2774177760375, -22.638715694779307, 87.43764488888883, -1.366299056227973]
        bump = []

        target_bump = [ 4.02587004e+00, -9.31306890e-05,  7.95482423e+00,  8.68548964e-03]
        for i in range(11):
            bump.append((0, [x*i/10. for x in target_bump]))
        self.find_bump_parameters["seed_errors"] = [1e-2]*10
        self.find_bump_parameters["closed_orbit"] = co
        self.find_bump_parameters["bump"] = bump
        self.find_bump_parameters["max_iterations"] = 1000
        self.find_bump_parameters["target_n_hits"] = 2
        self.find_bump_parameters["subs_overrides"] = {
            "__n_turns__":0.6,
            "__do_magnet_field_maps__":False,
            "__do_bump__":True,
        }
        self.find_bump_parameters["algorithm"] = "simplex"
        self.find_bump_parameters["staged_optimisation"][0] = { # get foil position
            "seed_fields":{
                        "__h_bump_1_field__":0.0,
                        "__v_bump_1_field__":0.0,
                        "__h_bump_2_field__":0.0,
                        "__v_bump_2_field__":0.0,
                        "__h_bump_3_field__":0.0,
                        "__v_bump_3_field__":0.1,
                        "__h_bump_4_field__":0.0,
                        "__v_bump_4_field__":0.0,
                        "__h_bump_5_field__":0.0,
                        "__v_bump_5_field__":0.0,
            },
            "target_orbit":dict([
                (0, foil_orbit),
                ]+[(i, co) for i in (1,)]),
            "position_tolerance":x_tol,
            "momentum_tolerance":p_tol,
            "fix_bumps":["__v_bump_-1_field__", "__h_bump_-1_field__",
                         "__v_bump_-2_field__", "__h_bump_-2_field__",
                         "__v_bump_3_field__", "__h_bump_3_field__",
                         "__v_bump_4_field__", "__h_bump_4_field__",
                         "__v_bump_5_field__", "__h_bump_5_field__",],
        }
        self.find_bump_parameters["staged_optimisation"][1] = { # recover closed orbit
            "seed_fields":{},
            "target_orbit":dict([ #maps station:target orbit
                ]+[(i, co) for i in (1, 5, 6)]),
            "position_tolerance":x_tol,
            "momentum_tolerance":p_tol,
            "fix_bumps":["__v_bump_1_field__", "__h_bump_1_field__",
                         "__v_bump_2_field__", "__h_bump_2_field__",
                         "__v_bump_3_field__", "__h_bump_3_field__",
                         "__v_bump_-4_field__", "__h_bump_-4_field__",
                         "__v_bump_-5_field__", "__h_bump_-5_field__",],
        }

        self.track_bump["input_file"] = "find_bump_parameters_0*.out"
        self.track_bump["injection_orbit"] = foil_orbit
        self.track_bump["foil_optimisation_stage"] = 0
        self.track_bump["field_optimisation_stage"] = 1
        self.track_bump["proton_orbit_station"] = 1
        self.track_bump["proton_orbit_phi"]= 0.
        self.track_bump["subs_overrides"]["__magnet_width__"] = 0.5

        self.find_closed_orbits["seed"] = [co]