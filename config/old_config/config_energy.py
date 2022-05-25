import os
from . import config_sector_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for i in range(10):
            self.substitution_list.append(config.get_baseline_substitution())
            self.substitution_list[-1]["__step_size__"] = 1e-4
            self.substitution_list[-1]["__energy__"] = i*1.0+3.0
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/sector_baseline/energy_scan_2")
        self.find_closed_orbits["max_iterations"] = 10
        self.run_control["find_closed_orbits_4d"] = True
        self.run_control["find_da"] = False