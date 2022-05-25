import os
from . import config_sector_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for bb_z in [20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0]:
            self.substitution_list.append(config.get_baseline_substitution())
            self.substitution_list[-1]["__step_size__"] = 0.001
            self.substitution_list[-1]["__bb_length__"] = bb_z
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/sector_baseline/bb_length_scan")
        self.run_control["find_da"] = False
        self.run_control["find_bump_parameters"] = False
