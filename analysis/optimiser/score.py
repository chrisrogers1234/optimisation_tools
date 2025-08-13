import copy
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.polynomial_fitter import PolynomialFitter

class Score(object):
    def __init__(self):
        pass

    def get_score(self):
        pass

    def print(self):
        pass

    def requires_transfer_matrix(self):
        return False

    def get_input_psv_list(self):
        raise NotImplementedError("Should be overridden")

    def save_score(self):
        return {}

    def setup_one(self, score_config):
        raise NotImplementedError("Should be overridden")

    def setup(self, config):
        for key in config:
            if key not in self.__dict__:
                print(self.__dict__.keys())
                raise KeyError(f"Did not recognise {key} while setting up score")
            self.__dict__[key] = config[key]

    @classmethod
    def setup_scores(cls, score_config_list):
        score_types = {
            "psv":PSVScore,
            "tune":TuneScore,
        }
        score = SumSquareScore(score_config_list)
        for score_config in score_config_list:
            ScoreType = score_types[score_config["type"]]
            a_score = ScoreType()
            a_score.setup(score_config)
            score.score_list.append(a_score)
        return score

class SumSquareScore(Score):
    def __init__(self, config):
        super(Score).__init__()
        self.score_list = []

    def get_score(self, hit_list_of_lists):
        score = sum([a_score.get_score(hit_list_of_lists)**2 for a_score in self.score_list])
        return score

    def get_input_psv_list(self):
        hit_list = []
        for score in self.score_list:
            hit_list += score.get_input_psv_list()
        return hit_list


class PSVScore(Score):
    def __init__(self):
        super(Score).__init__()
        self.type = None
        self.variables = []
        self.print_variables = []
        self.station = 0
        self.print_r = []
        self.tolerance = []
        self.input_psv = []
        self.fmt = "14.9g"

    @classmethod
    def number_of_scores(cls):
        return 4

    def requires_transfer_matrix(self):
        return False

    def print_hit(self, hit):
        for var in self.print_variables:
            print(f"{var} {hit[var]:8.4g}", end=" ")
        if self.print_variables:
            print(flush = True)


    def get_score(self, hit_list_of_lists):
        hit_list = hit_list_of_lists[0]
        [xs, pxs, ys, pys] = self.variables
        x_score, px_score, y_score, py_score = 0., 0., 0., 0.
        self.score = -1e9
        i_list = range(len(self.variables))
        for i, hit in enumerate(hit_list):
            if hit["station"] == self.station:
                self.print_hit(hit)
                self.r = [hit[self.variables[i]] - self.input_psv[i] for i in i_list]
                self.score = sum([(self.r[i]/self.tolerance[i])**2 for i in i_list if self.tolerance[i]])
                print("score:", self.score)
                break
        return self.score

    def print(self):
        print("Orbit", self.station, end=" | ")
        for r in self.print_r:
            print(format(r, self.fmt), end=" ")
        print(" | ", end="")
        if self.score == None:
            print("Ignored")
        else:
            for s in self.score:
                print(format(s, self.fmt), end=" ")
            print("|")

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
        return [copy.deepcopy(self.input_psv)]


class TuneScore(Score):
    def __init__(self, optimisation_def, config, optimisation):
        super(Score).__init__()
        DecoupledTransferMatrix.det_tolerance = 1.
        self.variables = config.optimisation["variables"]
        self.station = optimisation_def[0]
        parameters = optimisation_def[1]
        self.target_tune = None
        self.tune_tolerance = None
        if parameters:
            self.target_tune = parameters[0:2]
            self.tune_tolerance = parameters[2:4]
        self.i_list = [i for i in range(len(self.vars))]
        self.n_tunes = int(len(self.vars)/2)
        self.tunes = [0 for i in range(self.n_tunes)]
        self.orbit = [0 for var in self.vars]
        self.tm = numpy.array([[0 for var in self.vars] for var in self.vars])
        self.fmt = "14.9g"

    def requires_transfer_matrix(self):
        return True

    @classmethod
    def number_of_scores(cls):
        return 2

    def get_station_hits(self, hit_list_of_lists):
        source_hits = []
        station_hits = []
        for hit_list in hit_list_of_lists:
            source_hits.append(hit_list[0])
            for hit in hit_list:
                if hit["station"] == self.station:
                    station_hits.append(hit)
                    break
        return source_hits, station_hits

    def get_input_psv_list(self, hit_list):
        psv_list = []
        ref = hit_list[0]
        for hit in hit_list[1:]:
            psv_list.append([hit[var]-ref[var] for var in self.variables])
        return psv_list

    def get_tm(self, source_list, station_list):
        psv_in = self.get_psv_list(source_list)
        psv_out = self.get_psv_list(station_list)
        fitter = PolynomialFitter(len(self.vars))
        transfer_map = fitter.fit_transfer_map(psv_in, psv_out)
        transfer_map = numpy.array(transfer_map)[:, 1:]
        return transfer_map

    def get_orbit(self, station_list):
        if station_list:
            orbit = [station_list[0][var] for var in self.vars]
        else:
            orbit = [0 for var in self.vars]
        return orbit

    def get_tune(self, tm):
        cosmu = (tm[0,0]+tm[1,1])/2.0
        if abs(cosmu) > 1.0:
            tune = cosmu
        else:
            angle = math.acos(cosmu)
            # if M*(1, 0) < 0 then one cell puts a particle in bottom two
            # quadrants i.e. phase advance > pi
            if tm[1,0] > 0:
                angle = 2*math.pi - angle
            tune = angle/2.0/math.pi
            #print("Get tune cosmu", cosmu, "phi", angle, "x1", tm[1,0], "nu", tune)
        return tune

    def get_score(self, hit_list_of_lists):
        source_hits, station_hits = self.get_station_hits(hit_list_of_lists)
        try:
            self.orbit = self.get_orbit(station_hits)
            self.tm = self.get_tm(source_hits, station_hits)
            for i in range(self.n_tunes):
                sub_matrix = numpy.array(self.tm)[2*i:2*i+2, 2*i:2*i+2]
                self.tunes[i] = self.get_tune(sub_matrix)
            if self.target_tune:
                self.score = [((self.tunes[i]-self.target_tune[i])/self.tune_tolerance[i])**2 for i in range(self.n_tunes)]
            else:
                self.score = [0 for i in range(self.n_tunes)]
        except Exception:
            sys.excepthook(*sys.exc_info())
            self.score = [10, 10]
        return self.score

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

    def print(self):
        print("Tune  ", end="")
        print(self.station, end=" | ")
        for tune in self.tunes:
            print(format(tune, self.fmt), end=" ")
        print(" | ", end="")
        for s in self.score:
            print(format(s, self.fmt), end=" ")
        print("|")

    @classmethod
    def setup(cls, config, optimisation):
        if "target_tune" not in optimisation:
            return []
        score_list = []
        for item in sorted(optimisation["target_tune"].items()):
            score_list.append(TuneScore(item, config, optimisation))
        return score_list
