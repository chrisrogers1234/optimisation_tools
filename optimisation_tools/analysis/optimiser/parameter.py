import numbers

class Parameter:
    def __init__(self):
        self.name = ""
        self.subs_key = None # for substituting into the run control file
        self.variable_name = None # for messaging between optimisations
        self.seed = None
        self.lower_limit = None
        self.upper_limit = None
        self.error = 1e-3
        self.fixed = False
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
        if self.subs_key is not None:
            subs[self.subs_key] = self.current_value

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
        if self.variable_name is not None:
            return {
                self.variable_name:self.current_value,
            }
        else:
            return {}

    def store_data(self):
        data = {
            "name":self.name,
            "subs_key":self.subs_key,
            "variable_name":self.variable_name,
            "seed":self.seed,
            "lower_limit":self.lower_limit,
            "upper_limit":self.upper_limit,
            "error":self.error,
            "fixed":self.fixed,
            "current_value":self.current_value,
            "current_error":self.current_error,
        }
        return data
