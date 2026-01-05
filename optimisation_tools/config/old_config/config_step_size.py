import os
from . import config_jan_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for step in [0.005, 0.001, 0.0005, 0.0001]:
            self.substitution_list.append(config.get_baseline_substitution())
            self.substitution_list[-1]["__max_x_power__"] = 6
            self.substitution_list[-1]["__step_size__"] = step
        delta = 1.0
        self.run_control["output_dir"] = os.path.join(os.getcwd(),
                                    "output/jan/step_size_scan_"+str(delta))
        self.find_closed_orbits["deltas"] = [delta]*4

