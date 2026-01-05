import optimisation_tools.toy_model.longitudinal_model.rf_programme as rf_programme
import rf_programme.rf_programme

class ConstantBucket(rf_programme.rf_programme.RFProgramme):
    def __init__(self):
        self.v0 = 0.004
        self.f0 = 1./393.0

    def get_frequency(self, t):
        return self.f0

    def get_voltage(self, t):
        return self.v0
