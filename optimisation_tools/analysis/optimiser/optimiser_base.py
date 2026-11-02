import json
import copy
import sys

from optimisation_tools.opal_tracking import OpalTracking

class OptimiserBase:
    def __init__(self):
        self.name = "No name"
        self.message_dict = {"optimisation":{}}
        self.config = None
        self.optimiser_config = None
        self.parameter_list = []
        self.tracking = None
        self.score = None
        self.type = None
        self.allow_restart = True
        self.disable = False
        self.score_result = None
        self.maximum_final_score = sys.float_info.max # if score > max, fail the optimisation

    def setup(self, optimisation_config):
        pass

    def run_optimisation(self):
        pass

    def accept_optimisation(self):
        """
        Returns true if optimisation was successful, else False

        Run at the end of optimisation. Proceed to the next optimisation if True
        """
        if self.score_result is None:
            score_res = str(self.score_result)
        else:
            score_res = f"{self.score_result:8.4g}"
        print(f"Final score {score_res} compared to {self.maximum_final_score:8.4g}")
        return self.score_result is None or self.score_result <= self.maximum_final_score

    def teardown(self):
        pass

    ######### Some implementation/utility methods #############

    def process_parameters(self, x):
        index = 0
        for par in parameter_list:
            if par.fixed:
                continue
            par.current_value = x[index]
            index += 1

    def receive_output(self, post_dict):
        """
        Messaging protocol for messaging between parameters and scores and over
        different optimisation stages
        """
        self.message_dict.update(post_dict)
        for parameter in self.parameter_list:
            parameter.receive_output(post_dict)
        self.tracking.receive_output(post_dict)
        self.score.receive_output(post_dict)
        #self.score_result = post_dict["score_result"]

    def post_output(self):
        for parameter in self.parameter_list:
            self.message_dict.update(parameter.post_output())
        self.message_dict.update(self.score.post_output())
        self.message_dict["optimisation"][self.name] = self.store_data()
        return self.message_dict

    def store_data(self):
        summary = {
            "name":self.name,
            "parameter_list":[par.store_data() for par in self.parameter_list],
            "score":self.score.store_data(),
            "score_result":self.score_result,
            "config":optimiser_config
        }
        return summary