import os
from . import config_baseline

class Config(config_baseline.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        nsteps = 21
        a1 = {"__d_offset__":-1., "__f_offset__":+1.}
        a0 = {"__d_offset__":-0.154, "__f_offset__":+0.154}
        for i in range(0, nsteps):
            offset = 0.154 + 0.001*i
            self.substitution_list.append(config_baseline.get_baseline_substitution())
            for key in deltas:
                self.substitution_list[-1][key] = a0[key]+i*a1[key]
        for key in deltas:
            print("Offset scan", [item[key] for item in self.substitution_list])
        self.find_closed_orbits["final_subs_overrides"] = \
                                    self.find_closed_orbits["subs_overrides"]
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/offset")
