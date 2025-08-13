import os
import shutil
import sys
import importlib
import subprocess
import datetime
import json
import importlib

import numpy
import ROOT

import xboa.common

import optimisation_tools.analysis.find_closed_orbits_4d
import optimisation_tools.analysis.find_tune
import optimisation_tools.analysis.find_da
import optimisation_tools.analysis.optimisation
import optimisation_tools.analysis.find_bump_parameters
import optimisation_tools.analysis.build_bump_surrogate_model
import optimisation_tools.analysis.track_bump
import optimisation_tools.analysis.track_beam

from optimisation_tools.utils import utilities

MODULES = [
    ("find_closed_orbits_4d", optimisation_tools.analysis.find_closed_orbits_4d),
    ("find_tune", optimisation_tools.analysis.find_tune),
    ("find_da", optimisation_tools.analysis.find_da),
    ("find_bump_parameters", optimisation_tools.analysis.find_bump_parameters),
    ("build_bump_surrogate_model", optimisation_tools.analysis.build_bump_surrogate_model),
    ("track_bump", optimisation_tools.analysis.track_bump),
    ("track_beam", optimisation_tools.analysis.track_beam),
    ("optimisation", optimisation_tools.analysis.optimisation),
]

def output_dir(config, config_file_name):
    output_dir = config.run_control["output_dir"]
    if config.run_control["clean_output_dir"]:
        try:
            shutil.rmtree(output_dir)
        except OSError:
            pass
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    shutil.copy2(config_file_name, output_dir)
    git_string = "Time "+str(datetime.datetime.now())+"\n"
    try:
        git_string += subprocess.check_output(["git", "log", "-1"]).decode('unicode_escape')
        git_string += subprocess.check_output(["git", "status"]).decode('unicode_escape')
    except Exception:
        git_string += "Error calling git"
    with open(output_dir+"/git_status", "w") as fout:
        print(git_string, file=fout)
    with open(output_dir+"/subs_list.json", "w") as fout:
        print(json.dumps(config.substitution_list, indent=2), file=fout)


def master_substitutions(config):
    xboa.common.substitute(config.tracking["master_file"], config.tracking["lattice_file"], config.master_substitutions)

def main():
    config_file_name, config = utilities.get_config()
    output_dir(config, config_file_name)
    utilities.setup_gstyle()
    ROOT.gErrorIgnoreLevel = config.run_control["root_verbose"]
    if config.run_control["random_seed"] != None:
        numpy.random.seed(config.run_control["random_seed"])
    for name, module in MODULES:
        if config.run_control[name]:
            try:
                module.main(config)
            except Exception:
                if "fail_on_error" not in config.run_control or config.run_control["fail_on_error"]:
                    raise
                else:
                    sys.excepthook(*sys.exc_info())
    if "execute_modules" not in config.run_control:
        return
    for module_name in config.run_control["execute_modules"]:
        importlib.import_module(module_name)
        



    print("Finished with output in", config.run_control["output_dir"])

if __name__ == "__main__":
    try:
        main()
    except:
        raise
    finally:
        print('\033[0m')
              