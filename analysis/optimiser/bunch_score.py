import math
import numpy
import scipy.interpolate
import xboa.bunch
import xboa.common

from optimisation_tools.analysis.optimiser import score

class InputBeam:
    def __init__(self):
        pass

    def get_psv_list(self):
        return []

class TM2DBeam:
    def __init__(self):
        self.transfer_matrix = [{
            "transfer_matrix":None,
            "covariance_matrix":None,
            "variables":[],
            "emittance":-1.0,
            "geometric":False,
        }]
        self.pid = None
        self.closed_orbit_variable = None
        self.central_momentum = None
        self.number_of_particles = 0
        self.closed_orbit = []

    def build_psv_tm(self, tm):
        transfer_matrix = tm["transfer_matrix"]
        print("bunch_score.TM2DBeam TRANSFER MATRIX", tm)
        emittance = tm["emittance"]
        geometric = tm["geometric"]
        mass = xboa.common.pdg_pid_to_mass[abs(self.pid)]
        ellipse = xboa.bunch.Bunch.build_ellipse_from_transfer_matrix(
                                            transfer_matrix,
                                            emittance,
                                            tm["tm_momentum"],
                                            mass,
                                            geometric)
        print("In TM2DBeam.build_psv_tm TM in: \n", tm["transfer_matrix"])
        print("TM TM2DBeam.build_psv_tm V out:\n", ellipse)
        if "mean" in tm:
            mean = tm["mean"]
        else:
            mean = [0.0 for i in range(ellipse.shape[0])]
        psv_list = numpy.random.multivariate_normal(mean, ellipse, self.number_of_particles)
        return psv_list

    def build_psv_cov(self, tm):
        ellipse = numpy.array(tm["covariance_matrix"])
        if "mean" in tm:
            mean = tm["mean"]
        else:
            mean = [0.0 for i in range(ellipse.shape[0])]
        psv_list = numpy.random.multivariate_normal(mean, ellipse, self.number_of_particles)
        return psv_list

    def get_psv_list(self):
        variables = []
        psv_all = numpy.array([[] for i in range(self.number_of_particles)])
        for tm in self.transfer_matrix:
            variables += tm["variables"]
            if "covariance_matrix" in tm and tm["covariance_matrix"] is not None:
                a_psv_list = self.build_psv_cov(tm)
            else:
                a_psv_list = self.build_psv_tm(tm)
            psv_all = numpy.concatenate((psv_all, a_psv_list), 1)
        print("bunch_score generated psv with shape", psv_all.shape)
        print("    using variables", variables)
        psv_list = [dict([
                    (variables[col], psv_all[row][col]) for col in range(psv_all.shape[1])
                ]) for row in range(psv_all.shape[0])]
        psv_list = self.add_closed_orbit(psv_list)
        return psv_list

    def add_closed_orbit_fixed(self, psv_list):
        co_dict = self.closed_orbit[0]
        for key in co_dict:
            if key is self.closed_orbit_variable:
                continue
            for psv in psv_list:
                psv[key] += co_dict[key]
        return psv_list

    def add_closed_orbit_interpolation(self, psv_list):
        interpolators = {}
        self.closed_orbit = sorted(self.closed_orbit, key = lambda co: co[self.closed_orbit_variable])
        x = [co[self.closed_orbit_variable] for co in self.closed_orbit]
        interpolator_keys = [key for key in self.closed_orbit[0] if key != self.closed_orbit_variable]
        for key in interpolator_keys:
            y = [co[key] for co in self.closed_orbit]
            interpolators[key] = scipy.interpolate.CubicSpline(x, y)

        print("Interpolation co 0 in ", psv_list[0])
        for psv in psv_list:
            for key in interpolator_keys:
                delta = interpolators[key](psv[self.closed_orbit_variable])
                psv[key] += delta
        print("Interpolation co 0 out ", psv_list[0])
        return psv_list

    def add_closed_orbit(self, psv_list):
        for co_dict in self.closed_orbit:
            if sorted(co_dict.keys()) != sorted(self.closed_orbit[0].keys()):
                raise KeyError("Expected closed orbit keys to be the same for all entries")
            if self.closed_orbit_variable not in co_dict:
                raise KeyError(f"Couldnt find closed orbit variable {self.closed_orbit_variable} in {co_dict}")
        if len(self.closed_orbit) == 0:
            pass
        elif len(self.closed_orbit) == 1: # no interpolation required
            psv_list = self.add_closed_orbit_fixed(psv_list)
        else:
            psv_list = self.add_closed_orbit_interpolation(psv_list)
        return psv_list


    def setup(self, config):
        for item in config.keys():
            if item not in self.__dict__.keys():
                raise KeyError(f"Did not recognise configuration item {item}")
            self.__dict__[item] = config[item]

    def receive_output(self, output_dict: dict):
        self.receive_output_co(output_dict)
        self.receive_output_tm(output_dict)


    def receive_output_tm(self, output_dict):
        for a_tm in self.transfer_matrix:
            for key, value in a_tm.items():
                try: # check top level - do we have dict element like {"mean":"__replace_me__"}
                    if value in output_dict.keys():
                        a_tm[key] = output_dict[value]
                except TypeError:
                    pass
                try: # loop through next level
                    for i, item in enumerate(value):
                        if item in output_dict.keys():
                            value[i] = output_dict[item]
                except TypeError:
                    pass

    def receive_output_co(self, output_dict):
        for i, a_co in enumerate(self.closed_orbit):
            try: # an entire closed orbit entry is a substitution
                if a_co in output_dict.keys():
                    self.closed_orbit[i] = output_dict[a_co]
            except TypeError:
                pass
            for key, value in a_co.items():
                try: # one of the values in closed orbit is a substitution
                    if value in output_dict.keys():
                        a_co[key] = output_dict[value]
                except TypeError:
                    pass



class BunchScore(score.Score):
    def __init__(self):
        super().__init__()
        self.type = "bunch_score"
        self.station_in = 0
        self.station_out = 0
        self.input_beam = {}
        self.optimisation_parameter = ""
        self.optimisation_axes = []
        self.cut = ""


    def setup(self, config):
        super().setup(config)
        self.beam_gen = TM2DBeam()
        self.beam_gen.setup(self.input_beam)

    def process_hits(self, station, list_of_list_of_hits):
        # all hits with correct station
        station_hits = [
                [hit for hit in hit_list if hit["station"] == station] 
            for hit_list in list_of_list_of_hits]
        # take the first hit for each event, if one exists
        station_hits = [hit_list[0] for hit_list in station_hits if len(hit_list)]
        bunch = xboa.bunch.Bunch.new_from_hits(station_hits)
        return bunch

    def get_score(self, list_of_list_of_hits):
        self.bunch_in = self.process_hits(self.station_in, list_of_list_of_hits)
        self.bunch_out = self.process_hits(self.station_out, list_of_list_of_hits)
        optimisation_in = self.bunch_in.get(self.optimisation_parameter, self.optimisation_axes)
        optimisation_out = self.bunch_out.get(self.optimisation_parameter, self.optimisation_axes)
        score = optimisation_out/optimisation_in
        for name, axes in [("bunch_weight", []), ("emittance", ["x", "y"])]:
            v_in = self.bunch_in.get(name, axes)
            v_out = self.bunch_out.get(name, axes)
            print(f"{name} {axes} {v_in} {v_out}")

        print(f"BunchScore {self.optimisation_parameter} {self.optimisation_axes} "
              f"in  {optimisation_in} out {optimisation_out} score {score}")
        print(f"BunchScore {self.optimisation_parameter} {self.optimisation_axes} "
              f"in  {optimisation_in} out {optimisation_out} score {score}")
        return score

    def receive_output(self, output_dict):
        print("BunchScore Receiving output", output_dict)
        self.beam_gen.receive_output(output_dict)

    def get_input_psv_list(self):
        psv_list = self.beam_gen.get_psv_list()
        return psv_list


score.Score.score_types["bunch_score"] = BunchScore


def main():
    mu = math.pi/3
    smu = math.sin(mu)
    cmu = math.cos(mu)
    alpha = 1.0
    beta = 2.0
    gamma = (1+alpha**2)/beta
    M = numpy.array([[cmu+alpha*smu, beta*smu], [-gamma*smu, cmu-alpha*smu]])
    l, Q = numpy.linalg.eig(M)
    print("Matrix:\n", M)
    V = numpy.linalg.inv(numpy.array([[beta, -alpha], [-alpha, gamma]]))
    print("Invariant?\n", V)
    MT = numpy.transpose(M)
    Vout = numpy.dot(numpy.dot(MT, V), M)
    print("Check?\n", Vout)

if __name__ == "__main__":
    main()