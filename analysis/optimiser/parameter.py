import numbers

class Parameter:
    def __init__(self):
        self.name = ""
        self.key = ""
        self.seed = None
        self.lower_limit = 0
        self.upper_limit = 0
        self.error = 1e-3
        self.fixed = False
        self.minuit_index = -1
        self.current_value = None
        self.current_error = None

    @classmethod
    def setup(cls, parameter_dict):
        parameter = Parameter()
        parameter.setup_one(parameter_dict)
        return parameter

    def setup_one(self, parameter_dict):
        for key, value in parameter_dict.items():
            if key not in self.__dict__:
                raise KeyError(f"Did not recognise parameter {key}")
            self.__dict__[key] = value
        if self.current_error == None:
            self.current_value = self.seed
            self.current_error = self.error
        elif self.current_error > 0:
            self.seed = self.current_value
            self.error = self.current_error

    def update_to_subs(self, subs):
        subs[self.key] = self.current_value

    def receive_output(self, post_dict):
        if isinstance(self.seed, numbers.Number):
            return
        for key, value in post_dict.items():
            if self.seed == key:
                self.seed = value
                self.current_value = value
        if not isinstance(self.seed, numbers.Number):
            raise RuntimeError(f"Failed to receive {self.seed}")

    def post_output(self):
        return {
            self.key:self.current_value,
        }

