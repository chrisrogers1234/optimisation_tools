import bisect
import os
from . import config_jan_baseline
import plotting.plot_orbit as plot_orbit

def get_baseline_substitution():
    subs = config_jan_baseline.get_baseline_substitution()
    subs["__step_size__"] = 0.00005
    return subs


class Config(config_jan_baseline.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.find_closed_orbits["max_iterations"] = 10

