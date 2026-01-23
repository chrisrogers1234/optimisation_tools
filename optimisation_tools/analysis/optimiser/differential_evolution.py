import sys

import scipy
import scipy.optimize

import optimisation_tools.analysis.optimiser.optimiser_base as optimiser_base

class DifferentialEvolution(optimiser_base.OptimiserBase):
    def __init__(self):
        super().__init__()
        self.target_score = 1e9
        self.number_of_processes = 1

    def setup(self, optimiser_config):
        for key in optimiser_config:
            if key not in self.__dict__:
                raise KeyError("did not recognise", key)
            self.__dict__[key] = optimiser_config[key]

    def run_optimisation(self):
        seed_list = [par.seed for par in self.parameter_list]
        bound_list = [(par.lower_limit, par.upper_limit) for par in self.parameter_list]
        for i, bound in enumerate(bound_list):
            if bound[0] is None:
                bound = (-inf, bound[1])
            if bound[1] is None:
                bound = (bound[0], inf)
            bound_list[i] = bound
        try:
            scipy.optimize.differential_evolution(
                func = self.run_function,
                bounds = bound_list,
                workers = self.number_of_processes,
                updating = "deferred",
                x0 = seed_list,
                tol = self.target_score
            )
        except StopIteration:
            print("Reached maximum iteration", self.max_iterations, "without convergence\n\n")
        except Exception:
            sys.excepthook(*sys.exc_info())
            print("DifferentialEvolution failed")
            raise
        print("done DifferentialEvolution")

    def run_function(self, x):
        for i, parameter in enumerate(self.parameter_list):
            parameter.current_value = x[i]
        post_dict = self.post_output()
        self.receive_output(post_dict)
        hit_list_of_lists = self.tracking.track(self.parameter_list, self.score)
        a_score = self.score.get_score(hit_list_of_lists)
        print(f"Total score: {a_score:10.4g}")
        self.score_result = a_score
        return a_score
