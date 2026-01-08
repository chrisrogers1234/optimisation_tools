import numbers
import sys
import time
import copy

from optimisation_tools.opal_tracking import OpalTracking

class OptimiserBase:
    def __init__(self):
        self.name = "No name"
        self.message_dict = {}
        self.config = None
        self.parameter_list = []
        self.tracking = None
        self.score = None
        self.type = None

    def setup(self, optimisation_config):
        pass

    def run_optimisation(self):
        pass

    def teardown(self):
        pass

    ######### Some implementation/utility methods #############

    def get_filename_root(self):
        fname = self.config.run_control["output_dir"]+"/"
        fname += self.config.optimisation["output_file"]
        return fname

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

    def post_output(self):
        for parameter in self.parameter_list:
            self.message_dict.update(parameter.post_output())
        self.message_dict.update(self.score.post_output())
        return self.message_dict
