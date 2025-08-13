
class Parameter:
    def __init__(self):
        self.name = ""
        self.key = ""
        self.seed = 0
        self.lower_limit = 0
        self.upper_limit = 0
        self.error = 1e-3
        self.fixed = False
        self.minuit_index = -1
        self.current_value = -1
        self.current_error = -1

    @classmethod
    def setup(cls, parameter_dict):
        parameter = Parameter()
        parameter.setup_one(parameter_dict)
        return parameter

    def setup_one(self, parameter_dict):
        if self.current_error > 0:
            self.seed = self.current_value
            self.error = self.current_error
        for key, value in parameter_dict.items():
            if key not in self.__dict__:
                raise KeyError(f"Did not recognise parameter {key}")
            self.__dict__[key] = value

    def update_to_subs(self, subs):
        subs[self.key] = self.current_value

    def to_dict(self):
        return self.__dict__
