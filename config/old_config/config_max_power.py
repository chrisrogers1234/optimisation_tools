import os
from . import config_sector_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for max_power in range(8, 17, 2):
            self.substitution_list.append(config.get_baseline_substitution())
            self.substitution_list[-1]["__step_size__"] = 0.001
            self.substitution_list[-1]["__max_x_power__"] = max_power
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/sector_baseline/max_x_power")
