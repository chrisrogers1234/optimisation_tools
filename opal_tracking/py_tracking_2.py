import shutil
import sys
import os
import importlib.util
import sys
import time

import optimisation_tools.opal_tracking
from optimisation_tools.opal_tracking import OpalTracking
import xboa.hit

def import_my_lib(mod_file):
    """This is a bit dark. Better to import in run_one.py and then make sure that setup does reload stuff"""
    a_dir = os.path.split(mod_file)[0]
    sys.path.append(a_dir)
    mod_name = os.path.splitext(os.path.split(mod_file)[1])[0]
    my_module = importlib.__import__(mod_name)
    importlib.reload(my_module)
    sys.path.remove(a_dir)
    return my_module

class PyOpalTracking2(OpalTracking):
    def __init__(self, config, probes, reference_hit):
        self.config = config
        self.here = os.getcwd()
        self.beam_filename = config.tracking["beam_file_out"]
        self.ref = reference_hit
        self.clear_path = None
        self.setup()
        self.setup_output_filename(probes)
        self.name_dict = {}
        self._read_probes = self._read_ascii_probes
        self.overrides = {}

    def setup(self):
        self.do_tracking = True
        self.verbose = self.config.tracking["verbose"]
        self.pyopal_file = self.config.tracking["lattice_file_out"]
        if not isinstance(self.pyopal_file, str): # it is probably a list
        	self.pyopal_file = self.pyopal_file[0]
        self.run_dir = os.path.split(self.pyopal_file)[0]
        if not self.run_dir:
            self.run_dir = "./" 
        self.mod_name = os.path.basename(self.pyopal_file)
        self.mod_name = os.path.splitext(self.pyopal_file)[0]
        self.exec_module = None

    #list_of_hits, file_name, reference_hit, verbose
    def _tracking(self, list_of_hits):
        if self.verbose:
            print("Tracking in dir", os.getcwd())
        self.setup_dist_file(list_of_hits, self.beam_filename, self.ref, self.verbose)
        self.cleanup()
        self.exec_module = import_my_lib(self.pyopal_file)
        old_time = time.time()
        try:
            self.exec_module.main(self.config, self.overrides)
        except:
            sys.excepthook(*sys.exc_info())
            print("Tracking failed")
        if self.verbose:
            print("Ran for", time.time() - old_time, "s")
            print(self.pyopal_file, os.getcwd())

    def track_many(self, list_of_hits):
        list_of_list_of_hits = super().track_many(list_of_hits)
        return list_of_list_of_hits[1:]

def setup_py_tracking_2(config, probes, reference_hit):
    tracking = PyOpalTracking2(config, probes, reference_hit)
    return tracking

class ConfigMockup():
    def __init__(self):
        self.substitution_list = [{
            "__lattice_phi_init__":0,
            "__energy__":3,
            "__bf__":-0.80,
            "__fd_offset__":0.0,
            "__do_magnet_field_maps__":False,
        }]
        self.tracking = {
            "lattice_file_out":["output/test/run_fets_ffa.py", "analytical_field.py"],
            "beam_file_out":"disttest.dat",
            "verbose":0,
        }
        self.tracking_log = {}

def test_main():
    shutil.copy("lattice/fets_ffa.in", "output/test/run_fets_ffa.py")
    config = ConfigMockup()
    probes = []
    reference_hit = xboa.hit.Hit()
    tracking = setup_py_tracking_2(config, probes, reference_hit)
    tracking.track_many([xboa.hit.Hit()])

if __name__ == "__main__":
    test_main()

