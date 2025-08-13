import os
import copy

from optimisation_tools.utils import utilities

class TrackingWrapper:
    def __init__(self):
        self.config = None
        self.subs = {}
        self.probe_files = []
        self.ref_energy = 0.0
        self.tmp_dir = ""
        self.tracking = None
        self.mass_shell_condition = ""

    def setup(self, config, tracking_config):
        self.config = config
        for key in tracking_config:
            if key not in self.__dict__:
                raise KeyError("Did not recognise parameter", key)
            self.__dict__[key] = tracking_config[key]
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"], self.tmp_dir)

    def setup_subs(self, parameter_list, hit_list):
        subs = self.config.substitution_list[0]
        subs.update(self.subs)
        for par in parameter_list:
            par.update_to_subs(subs)
            print("    ", par.name, par.current_value)
        utilities.do_lattice(self.config, subs, {}, hit_list, self.tracking)

    def track(self, parameter_list, score):
        hit_list = score.get_input_psv_list()
        hit_list_of_lists = self.track_some(parameter_list, hit_list)
        return hit_list_of_lists

    def generate_hit_list(self, psv_list):
        hit_list = []
        for psv in psv_list:
            my_hit = copy.deepcopy(self.tracking.ref)
            my_hit.mass_shell_condition(self.mass_shell_condition)
            hit_list.append(my_hit)
        return hit_list

    def track_some(self, parameter_list, psv_list):
        utilities.clear_dir(self.tmp_dir)
        os.chdir(self.tmp_dir)
        # fix momentum
        self.tracking = utilities.setup_tracking(self.config, self.probe_files, self.ref_energy)
        hit_list = self.generate_hit_list(psv_list)
        self.setup_subs(parameter_list, hit_list)
        #print("Reference kinetic energy:", self.tracking.ref["kinetic_energy"])
        #print("Seed kinetic energy:     ", [hit["kinetic_energy"] for hit in hit_list], flush=True)
        hit_list_of_lists = self.tracking.track_many(hit_list)
        #print("Station to probe mapping:\n   ", end=' ')
        #for i, fname in enumerate(self.tracking.get_name_list()):
        #    print("("+str(i)+",", fname+")", end=' ')
        #print()
        return hit_list_of_lists