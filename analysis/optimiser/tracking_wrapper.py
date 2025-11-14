import numbers
import os
import copy

from optimisation_tools.utils import utilities

class TrackingWrapper:
    def __init__(self):
        self.config = None
        self.subs = {} # subs from user input
        self.received_subs = {} # subs after receiving; this needs to be re-updated on every iteration
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

    def track(self, parameter_list, score):
        psv_list = score.get_input_psv_list()
        hit_list_of_lists = self.track_some(parameter_list, psv_list)
        return hit_list_of_lists

    def receive_output(self, post_dict):
        self.received_subs = copy.deepcopy(self.subs)
        for key, value in self.received_subs.items():
            if isinstance(value, numbers.Number):
                continue
            baseline_value = self.config.substitution_list[0][key]
            if value in post_dict.keys():
                print(f"Replacing '{key}' '{self.received_subs[key]}' with '{post_dict[value]}'")
                self.received_subs[key] = post_dict[value]
            elif isinstance(baseline_value, numbers.Number):
                # the current value is not a number, but the baseline is a number
                # maybe this should have been replaced?
                print(f"Warning: not replacing substitution '{key}' with value '{self.received_subs[key]}' and baseline value was {baseline_value}")


    def generate_hit_list(self, psv_list):
        hit_list = []
        for i, psv in enumerate(psv_list):
            my_hit = copy.deepcopy(self.tracking.ref)
            for key, value in psv.items():
                my_hit[key] = value
            my_hit.mass_shell_condition(self.mass_shell_condition)
            my_hit["event_number"] = i
            hit_list.append(my_hit)
        return hit_list

    def setup_subs(self, parameter_list, hit_list):
        subs = self.config.substitution_list[0]
        subs.update(self.received_subs)
        for par in parameter_list:
            par.update_to_subs(subs)
            print("    ", par.name, par.current_value)
        utilities.do_lattice(self.config, subs, {}, hit_list, self.tracking)

    def track_some(self, parameter_list, psv_list):
        utilities.clear_dir(self.tmp_dir)
        os.chdir(self.tmp_dir)
        # fix momentum
        self.tracking = utilities.setup_tracking(self.config, self.probe_files, self.ref_energy)
        hit_list = self.generate_hit_list(psv_list)
        self.setup_subs(parameter_list, hit_list)
        hit_list_of_lists = self.tracking.track_many(hit_list)
        return hit_list_of_lists