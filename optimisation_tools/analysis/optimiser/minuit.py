import ctypes

try:
    import ROOT
except:
    print("Optimiser needs ROOT")

import optimisation_tools.analysis.optimiser.optimiser_base as optimiser_base

def minuit_function(*args):
    try:
        MinuitOptimiser.minuit_global.minuit_function(*args)
    except Exception:
        print("Score function failed")
        sys.excepthook(*sys.exc_info())
        raise


class Minuit(optimiser_base.OptimiserBase):
    def __init__(self):
        super(Optimiser).__init__()
        self.algorithm = "Migrad"
        self.target_score = 1e9
        self.max_iterations = 1000
        self.iteration = 0
        self.minuit = None
        self.iteration_number = 0

    def setup(self, optimiser_config):
        self.minuit = ROOT.TMinuit()
        self.minuit.SetFCN(minuit_function)
        for key in optimiser_config:
            if key not in self.__dict__:
                raise KeyError("did not recognise", key)
            self.__dict__[key] = optimiser_config[key]

    def setup_parameter(self, index):
        parameter = self.parameter_list[index]
        self.minuit.DefineParameter(index, parameter.name,
                               parameter.seed, abs(parameter.error),
                               parameter.lower_limit, parameter.upper_limit)
        if parameter.fixed:
            self.minuit.FixParameter(index)
        parameter.index = index

    def prerun_setup(self):
        for i, parameter in enumerate(self.parameter_list):
            self.setup_parameter(i)
        self.iteration_number = 0
        MinuitOptimiser.minuit_global = self

    def update_parameter(self, index):
        value = ctypes.c_double()
        error = ctypes.c_double()
        self.minuit.GetParameter(index, value, error)
        self.parameter_list[index].current_value = value.value
        self.parameter_list[index].current_error = error.value

    def run_optimisation(self):
        print("setup minuit")
        self.prerun_setup()
        print("run minuit")
        try:
            self.minuit.Command(self.algorithm+" "+str(self.max_iterations)+" "+str(self.target_score))
        except StopIteration:
            print("Reached maximum iteration", self.max_iterations, "without convergence\n\n")
        except Exception:
            sys.excepthook(*sys.exc_info())
            print("Minuit failed")
            raise
        print("done minuit")

    def teardown(self):
        pass


    def update_parameters_from_minuit(self):
        for i, parameter in enumerate(self.parameter_list):
            parameter.update_from_minuit(i)

    def get_score(self, hit_list_of_lists):
        total_score = self.score.get_score()
        return total_score

    def minuit_function(self, nvar=None, parameters=None, score=ctypes.c_float(0.0), jacobian=None, err=None):
        self.iteration_number += 1
        if self.iteration_number > self.max_iterations:
            raise StopIteration("Run out of iterations")
        for i, parameter in enumerate(self.parameter_list):
            self.update_parameter(i)
        post_dict = self.post_output()
        self.receive_output(post_dict)
        hit_list_of_lists = self.tracking.track(self.parameter_list, self.score)
        a_score = self.score.get_score(hit_list_of_lists)
        score.value =  a_score
        print(f"Total score: {a_score:10.4g}  Iteration {self.iteration_number}/{self.max_iterations}\n")

    root_algorithms = ["simplex", "migrad"]
    minuit_global = None