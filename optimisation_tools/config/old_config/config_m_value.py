import os
from . import config_baseline

class Config(config_baseline.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        nsteps = 51
        delta = 0.5
        for i in range(0, nsteps): #range(0, -nsteps, -1)+
            m_value = 1.58 - delta/2 + delta*i/float(nsteps-1)
            self.substitution_list.append(config_baseline.get_baseline_substitution())
            self.substitution_list[-1]["__m_index__"] = m_value
        print("M scan", [item["__m_index__"] for item in self.substitution_list])
        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/m_scan")
