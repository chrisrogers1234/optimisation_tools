
class RFProgramme(object):
    def __init__(self):
        pass

    def get_frequency(self, t):
        raise NotImplementedError()

    def get_voltage(self, t):
        raise NotImplementedError()

    def get_frequency_list(self, t_list):
        return [self.get_frequency(t) for t in t_list]

    def get_voltage_list(self, t_list):
        return [self.get_voltage(t) for t in t_list]
