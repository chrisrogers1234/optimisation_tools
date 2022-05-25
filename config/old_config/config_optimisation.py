import os
from . import config_triplet_baseline as config

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        for i in range(6):
            end_length = 0.15
            bf = 0.5
            bd = -0.24
            tilt = 0.0
            l_f = 0.40
            l_d = 0.24
            drift = 1.3
            m_index = 1.46+i*0.02
            self.substitution_list.append(config.get_baseline_substitution())
            self.substitution_list[-1]["__step_size__"] = 1e-3
            self.substitution_list[-1]["__f_tilt_angle__"] = tilt
            self.substitution_list[-1]["__bf__"] = -bf/2*(l_f/0.4)/(l_d/0.24)
            self.substitution_list[-1]["__bd__"] = bd/2.
            self.substitution_list[-1]["__f_length__"] = l_f
            self.substitution_list[-1]["__d_length__"] = l_d
            self.substitution_list[-1]["__f_end_length__"] = end_length
            self.substitution_list[-1]["__d_end_length__"] = end_length
            self.substitution_list[-1]["__fd_gap__"] = (2.5-l_d*1-l_f*2-drift)/2.
            self.substitution_list[-1]["__magnet_width__"] = 1.0
            self.substitution_list[-1]["__m_index__"] = m_index
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/triplet_baseline/optimisation_scratch")
        self.run_control["find_closed_orbits_4d"] = False
        self.run_control["find_da"] = True
        self.find_closed_orbits["max_iterations"] = 10
        self.find_closed_orbits["seed"] = [ [3742.7470664498355, -0.00012571620902690483, -34.44270479613738, 5.8877333557916245e-05]]
