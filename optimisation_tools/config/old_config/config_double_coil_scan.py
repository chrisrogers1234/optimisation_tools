import os
from . import config_double_triplet_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for i in range(11):
            delta_bf = (i)*0.2
            delta_bd = (i)*0.2
            self.substitution_list.append(config.get_baseline_substitution())
            self.substitution_list[-1]["__bf_offset__"] = 0.02
            self.substitution_list[-1]["__bd_offset__"] = 0.02
            self.substitution_list[-1]["__delta_bf__"] = delta_bf
            self.substitution_list[-1]["__delta_bd__"] = delta_bd
        self.run_control = {
            "find_closed_orbits_4d":True,
            "find_tune":False,
            "find_da":False,
            "find_bump_parameters":False,
            "track_bump":False,
            "build_bump_surrogate_model":False,
            "clean_output_dir":False,
            "output_dir":os.path.join(os.getcwd(), "output/double_triplet_baseline/double_coil_scan"),
            "root_verbose":6000,
        }
