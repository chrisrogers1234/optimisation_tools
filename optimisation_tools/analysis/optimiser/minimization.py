import copy
import sys

import scipy
import scipy.optimize

import optimisation_tools.analysis.optimiser.optimiser_base as optimiser_base

class Minimization(optimiser_base.OptimiserBase):
    def __init__(self):
        super().__init__()

        self.method = "nelder-mead"
        self.target_score = 1e9
        self.max_iterations = 1000
        self.iteration = 0
        self.fixed_parameter_list = []
        self.var_parameter_list = []

    def setup(self, optimiser_config):
        for key in optimiser_config:
            if key not in self.__dict__:
                raise KeyError("did not recognise", key)
            self.__dict__[key] = optimiser_config[key]

    def run_optimisation(self):
        self.var_parameter_list = [par for par in self.parameter_list if not par.fixed]
        self.fixed_parameter_list = [par for par in self.parameter_list if par.fixed]
        if len(self.var_parameter_list) == 0:
            raise ValueError("At least one parameter must be defined that is not fixed")

        bound_list = [(par.lower_limit, par.upper_limit) for par in self.var_parameter_list]
        seed_list = [par.seed for par in self.var_parameter_list]
        options = {"maxiter":self.max_iterations}
        if self.method == "nelder-mead":
            options["initial_simplex"] = self.initial_simplex()

        for i, bound in enumerate(bound_list):
            if bound[0] is None:
                bound = (-inf, bound[1])
            if bound[1] is None:
                bound = (bound[0], inf)
            bound_list[i] = bound
        try:
            scipy.optimize.minimize(
                fun = self.run_function,
                x0 = seed_list,
                bounds = bound_list,
                tol = self.target_score,
                options = options,
                method = self.method,
            )
        except StopIteration:
            print("Reached maximum iteration", self.max_iterations, "without convergence\n\n")
            print("(Minimization) self.score_result", self.score_result)
        except Exception:
            sys.excepthook(*sys.exc_info())
            raise

    def initial_simplex(self):
        x0 = [par.seed for par in self.var_parameter_list]
        simplex = [x0]
        for i, par in enumerate(self.var_parameter_list):
            if par.fixed:
                continue
            xi = copy.deepcopy(x0)
            xi[i] += par.error
            simplex.append(xi)
        return simplex

    def run_function(self, x):
        self.iteration += 1
        for i, parameter in enumerate(self.var_parameter_list):
            parameter.current_value = x[i]
        post_dict = self.post_output()
        self.receive_output(post_dict)
        hit_list_of_lists = self.tracking.track(self.var_parameter_list+self.fixed_parameter_list, self.score)
        a_score = self.score.get_score(hit_list_of_lists)
        self.score_result = a_score
        print(f"Iteration: {self.iteration:04d} Total score: {a_score:10.4g} result: {self.score_result}\n")
        if self.iteration >= self.max_iterations:
            raise StopIteration("No more iterations")
        return a_score
