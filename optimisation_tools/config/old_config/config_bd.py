import os
from . import config_baseline

class Config(config_baseline.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        nsteps = 11
        delta = 0.1
        for i in list(range(0, -nsteps, -1))+list(range(1, nsteps)):
            bd = 0.25 - delta*i/float(nsteps-1)
            bf = -0.25/0.44 + delta*i/float(nsteps-1)
            self.substitution_list.append(config_baseline.get_baseline_substitution())
            self.substitution_list[-1]["__m_index__"] = 1.7
            self.substitution_list[-1]["__bd__"] = bd
            self.substitution_list[-1]["__bf__"] = bf
            self.substitution_list[-1]["__b_ratio__"] = -bd/bf
        print("BD scan", [item["__bd__"] for item in self.substitution_list])
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/b_scan")
