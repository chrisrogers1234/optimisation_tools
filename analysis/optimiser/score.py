import sys
import numbers
import copy

import numpy
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.polynomial_fitter import PolynomialFitter

class Score(object):
    def __init__(self):
        pass

    def get_score(self, hit_list_of_lists):
        raise NotImplementedError("Should be overridden")

    def receive_output(self, post_dict):
        raise NotImplementedError("Should be overridden")

    def get_input_psv_list(self):
        raise NotImplementedError("Should be overridden")

    def save_score(self):
        raise NotImplementedError("Should be overridden")

    def post_output(self): # optionally, we post output from the score function
        return {}

    def setup(self, config):
        for key in config:
            if key not in self.__dict__:
                print(self.__dict__.keys())
                raise KeyError(f"Did not recognise {key} while setting up score")
            self.__dict__[key] = config[key]

    @classmethod
    def setup_scores(cls, score_config_list):
        if isinstance(score_config_list, Score):
            score = score_config_list
            return score
        score = SumSquareScore(score_config_list)
        for score_config in score_config_list:
            ScoreType = cls.score_types[score_config["type"]]
            a_score = ScoreType()
            a_score.setup(score_config)
            score.score_list.append(a_score)
        return score

    @classmethod
    def reconcile_dict(cls, parameter_dict, psv):
        """
        Substitutes values from parameter_dict into psv

        - psv: psv that will be reconciled
        - parameter_dict: dict of inputs to use for reconciling

        Searches through psv and look for value that are not numeric. For each
        value, try to find the value as a key in parameter_dict. If found,
        replace the value in psv with the value in parameter_dict.

        The idea is that parameter_dict is a dict of values that is passed
        between optimisations and used to substitute PSVs.
        """
        psv = copy.deepcopy(psv)
        for psv_key, psv_value in psv.items():
            if isinstance(psv_value, numbers.Number):
                continue
            for key, value in parameter_dict.items():
                if key == psv_value:
                    psv[psv_key] = value
        return psv

    score_types = {}


class SumSquareScore(Score):
    def __init__(self, config):
        """
        Calculate score based on sum of the square of several other scores

        - score_list: the list of scores
        - score_index_list: mapping from the full psv list to the psv list for a
        particular score, so that the correct hits can be handed back for score
        calculation
        """
        super(Score).__init__()
        # not updated every iteration:
        self.score_list = []
        # updated every iteration:
        self.psv_list = []
        self.score_index_list = []

    def get_score(self, hit_list_of_lists):
        score_output_list = []
        for i, score in enumerate(self.score_list):
            a_hit_list_of_lists = [hit_list_of_lists[psv_index] for psv_index in self.score_index_list[i]]
            score_output_list.append(score.get_score(a_hit_list_of_lists))
        score = sum([a_score**2 for a_score in score_output_list])
        return score

    def compact_psv_list(self):
        """
        Remove duplicates
        """
        compacted_psv_list = []
        for original_i, psv in enumerate(self.psv_list):
            if psv in compacted_psv_list:
                new_i = compacted_psv_list.index(psv)
            else:
                compacted_psv_list.append(psv)
                new_i = len(compacted_psv_list)-1
            for index_list in self.score_index_list:
                for j, item in enumerate(index_list):
                    if index_list[j] == original_i:
                        index_list[j] = new_i
        self.psv_list = compacted_psv_list

    def receive_output(self, post_dict):
        for score in self.score_list:
            score.receive_output(post_dict)

    def get_input_psv_list(self):
        self.psv_list = []
        self.score_index_list = [None]*len(self.score_list)
        for i, score in enumerate(self.score_list):
            score_psv_list = score.get_input_psv_list()
            self.score_index_list[i] = [i for i in range(len(self.psv_list), len(self.psv_list)+len(score_psv_list))]
            self.psv_list += score_psv_list
        self.compact_psv_list()
        for i, psv in enumerate(self.psv_list):
            if i < 3 or len(self.psv_list)-i < 4:
                print(psv)
            elif i == 3:
                print(f"<omitting {len(self.psv_list)-6} phase space vectors>")
        return self.psv_list

    def post_output(self):
        post_dict = {}
        for score in self.score_list:
            post_dict.update(score.post_output())
        return post_dict

class PSVScore(Score):
    def __init__(self):
        super(Score).__init__()
        self.type = None
        self.print_variables = []
        self.station = 0
        self.tolerance = {}
        self.input_psv = {}
        self.reference_psv = {}
        self.post_key = ""
        # updated every iteration, for example if one of the values is parameter
        # dependent; e stands for "ephemeral"
        self.e_input_psv = {}
        self.e_ref_psv = {}
        self.verbose = False

    def print_hit(self, hit):
        for var in self.print_variables:
            print(f"{var} {hit[var]:8.4g}", end=" ")
        if self.print_variables:
            print(flush = True)

    def receive_output(self, parameter_list):
        self.e_input_psv = Score.reconcile_dict(parameter_list, self.input_psv)
        self.e_ref_psv = Score.reconcile_dict(parameter_list, self.reference_psv)
        if self.verbose:
            print("PSVScore function received output", parameter_list, "\n to yield", self.e_input_psv, self.e_ref_psv)

    def get_score(self, hit_list_of_lists):
        hit_list = hit_list_of_lists[0]
        self.score = -1e9
        for i, hit in enumerate(hit_list):
            if hit["station"] == self.station:
                self.print_hit(hit)
                self.r = dict([(var, hit[var] - self.e_ref_psv[var]) for var in self.e_ref_psv.keys()])
                self.score = sum([(self.r[var]/self.tolerance[var])**2 for var in self.e_ref_psv.keys() if self.tolerance[var]])
                print("score:", self.score)
                break
        return self.score

    def post_output(self):
        post_dict = {}
        if self.post_key != "":
            post_dict[self.post_key] = self.e_ref_psv
        return post_dict

    def save_score(self):
        output = {
            "score_type":"orbit",
            "station":self.station,
            "target":self.target_co,
            "orbit":self.print_r,
            "weighted_orbit":self.score
        }
        return output

    def get_input_psv_list(self):
        return [copy.deepcopy(self.e_input_psv)]

class TMScore(Score):
    def __init__(self):
        super(Score).__init__()
        self.type = "transfer_matrix"
        self.central_psv = {}
        self.deltas = {}
        self.source_station = 0
        self.target_station = None
        self.tm_variables = []
        self.post_key = ""
        self.tm = None
        # ephemeral data
        self.e_central_psv = {}
        self.e_deltas = {}

    def receive_output(self, output_dict):
        self.e_central_psv = Score.reconcile_dict(output_dict, self.central_psv)
        self.e_deltas = Score.reconcile_dict(output_dict, self.deltas)

    def get_station_hits(self, hit_list_of_lists):
        source_hits = []
        station_hits = []
        for hit_list in hit_list_of_lists:
            source_hits.append(hit_list[self.source_station])
            for hit in hit_list:
                if hit["station"] == self.target_station:
                    station_hits.append(hit)
                    break
        return source_hits, station_hits

    def get_input_psv_list(self):
        print("Getting psvs for transfer matrix calculation")
        ref = copy.deepcopy(self.e_central_psv)
        psv_list = [ref]
        for key, value in self.e_deltas.items():
            for d in -1, +1:
                hit = copy.deepcopy(ref)
                print("    Setting key", key, "value", value, "ref", hit[key])
                hit[key] += d*value
                psv_list.append(hit)
        return psv_list

    def get_psv_list(self, hit_list):
        psv_list = []
        for hit in hit_list:
            psv_list.append([hit[key] for key in self.tm_variables])
        return psv_list

    def get_tm(self, psv_in, psv_out):
        fitter = PolynomialFitter(len(self.deltas))
        transfer_map = fitter.fit_transfer_map(psv_in, psv_out)
        transfer_map = numpy.array(transfer_map)[:, 1:]
        return transfer_map

    def get_score(self, hit_list_of_lists):
        source_list, station_list = self.get_station_hits(hit_list_of_lists)
        if len(source_list) != len(station_list):
            return -1e9
        psv_in = self.get_psv_list(source_list)
        psv_out = self.get_psv_list(station_list)
        for i, psv_i in enumerate(psv_in):
            for value in psv_i:
                sys.stdout.write(f"{value:10.6f} ")
            sys.stdout.write("| ")
            for value in psv_out[i]:
                sys.stdout.write(f"{value:10.6f} ")
            print()
        self.tm = self.get_tm(psv_in, psv_out)
        print("Transfer matrix:\n"+str(self.tm)+"\n")
        return 0.0

    def post_output(self):
        post_dict = {}
        if self.post_key != "":
            post_dict[self.post_key] = self.tm
        print("TMSCore posting output", post_dict)
        return post_dict

    def save_score(self):
        output = {
            "score_type":"tune",
            "station":self.station,
            "orbit":self.orbit,
            "tunes":self.tunes,
            "target":self.target_tune,
            "transfer_matrix":self.tm.tolist(),
            "weighted_tunes":self.score
        }
        return output

Score.score_types = {
    "psv":PSVScore,
    "transfer_matrix":TMScore,
}
