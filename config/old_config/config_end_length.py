import os
from . import config_triplet_baseline

class Config(config_triplet_baseline.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for i in range(6):
            length = 0.20-i*0.02
            self.substitution_list.append(config_triplet_baseline.get_baseline_substitution())
            self.substitution_list[-1]["__d_end_length__"] = length
            self.substitution_list[-1]["__f_end_length__"] = length
            self.substitution_list[-1]["__step_size__"] = 1e-4
        print("End length scan", [item["__f_end_length__"] for item in self.substitution_list])
        self.run_control = {
            "find_closed_orbits_4d":True,
            "find_tune":False,
            "find_da":False,
            "find_bump_parameters":False,
            "track_bump":False,
            "clean_output_dir":False,
            "output_dir":os.path.join(os.getcwd(), "output/triplet_baseline/end_length_scan"),
            "root_verbose":6000,
        }
