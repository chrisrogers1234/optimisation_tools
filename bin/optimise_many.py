import copy
import subprocess
import os
import time
import sys

import scipy
import scipy.optimize

from optimisation_tools.utils import utilities

class OptimiseMany:
    def __init__(self):
        self.config_dir = "config/"
        self.config = None
        self.exe = ["sleep", "1"] # list containing executable default arguments
        self.parameter_list = [] # list of parameters
        self.unique_run_id = 0

    def load_configs(self):
        config_file_name, self.config = utilities.get_config(self.config_dir)
        print("Loading configs from ", config_file_name)
        self.parameter_list = self.config.get_parameters()
        self.exe = self.config.get_script()
        print(f"Found exe {self.exe}")

    def run_item(self, x):
        command_line_args = self.exe+x
        subprocess.POpen(command_line_args)

    def run_de(self):
        maxiter = self.config.get_optimisation_control()["maximum_iteration"]
        atol = self.config.get_optimisation_control()["absolute_tolerance"]
        disp = self.config.get_optimisation_control()["verbose"]
        workers = self.config.get_optimisation_control()["workers"]
        will_polish = self.config.get_optimisation_control()["will_polish"]
        bounds = []
        seed = []
        for param in self.parameter_list:
            if param["fixed"]:
                continue # we don't give DE the fixed parameters -  we add at func time
            bounds.append((param["min"], param["max"]))
            seed.append(param["seed"])


        result = scipy.optimize.differential_evolution(
            self.run_function,
            bounds,
            maxiter=maxiter, # maximum tolerance
            atol=atol, # absolute tolerance
            disp=disp, # verbose
            workers=workers, # number of concurrent jobs
            x0=seed, # seed values
            polish=will_polish, # do some post optimisation?
            updating="deferred" # keep running job even if previous job not finished
        )
        print("Minimisation returns", result)

    def run_function(self, x_value):
        x_index = 0
        func_call = copy.deepcopy(self.exe)
        for param in self.parameter_list:
            if param["fixed"]:
                func_call.append(param["seed"])
            else:
                func_call.append(x_value[x_index])
                x_index += 1
        func_call = [str(value) for value in func_call]
        with open(self.config.get_log_filename(func_call), "w") as log_file:
            proc = subprocess.Popen(func_call, stdout=log_file, stderr=subprocess.STDOUT)
            proc.wait()
        print("Function call", func_call, "pid", proc.pid, "finished with return code", proc.returncode)
        score = self.config.get_score(func_call)
        print("Got score", score)
        sys.stdout.flush()
        time.sleep(1)
        return score

class ConfigBase:
    """
    ConfigBase base class that tells OptimiseMany how to control jobs

    ConfigBase is a pure abstract base class. It continas
    """
    def __init__(self):
        pass

    def get_optimisation_control(self):
        """
        Return control dict containing control parameters
        """
        raise NotImplementedError("Need to overload this method")

    def get_score(self, func_call):
        """
        return a float that is the score from a particular job
        """
        raise NotImplementedError("Need to overload this method")

    def get_log_filename(self, func_call):
        """
        return a filename where logging will be done
        """
        raise NotImplementedError("Need to overload this method")

def main():
    optimiser = OptimiseMany()
    optimiser.load_configs()
    optimiser.run_de()

if __name__ == "__main__":
    main()
