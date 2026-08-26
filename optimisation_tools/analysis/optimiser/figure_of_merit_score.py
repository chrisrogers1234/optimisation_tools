"""
Return score following tracking a full beam, corresponding to a figure of merit

Figure of merit is
    (transverse emittance)*(longitudinal emittance)/transmission**2

Tracked beam sets up using dispersion, amplitude momentum coefficient, etc
"""
import operator

import numpy

import xboa
import xboa.common
import xboa.bunch

import optimisation_tools.analysis.optimiser.score

class FigureOfMeritScore(optimisation_tools.analysis.optimiser.score.Score):
    def __init__(self, beam_file, station, n_particles, pz, transverse_emittance, longitudinal_emittance, do_scan, a_pz_coefficient):
        self.beam_file = beam_file
        self.station = station
        self.file_format = "icool_for009"
        self.p_mean = pz
        self.eps_cut_long = 1000
        self.n_particles = n_particles
        self.eps_cut_trans = 1000
        if self.n_particles % 2 == 0:
            self.n_particles += 1
        self.emittance_trans = transverse_emittance
        self.frequency = 0.704 # GHz
        self.emittance_long = longitudinal_emittance
        self.dispersion_dp = 2.0 # MeV/c
        self.mass = xboa.common.pdg_pid_to_mass[13]
        self.variables = ["x", "px", "y", "py", "t", "energy"]
        self.do_scan = do_scan
        self.bunch = None
        self.a_pz_coefficient = a_pz_coefficient
        self.beam_x_p_plus = None
        self.beam_y_p_plus = None
        self.beam_x_p_minus = None
        self.beam_y_p_minus = None
        self.cov_trans = None # target cov matrix
        self.station_us = 4
        self.station_ds = 19
        self.output = {}

    def apply_cuts(self):
        self.bunch.cut({"pid":-13}, operator.ne)

    def receive_output(self, post_dict):
        self.beam_x_p_plus = post_dict["$beam_x+p"] # delta to off-momentum orbit
        self.beam_y_p_plus = post_dict["$beam_y+p"] # delta to off-momentum orbit
        self.beam_x_p_minus = post_dict["$beam_x-p"] # delta to off-momentum orbit
        self.beam_y_p_minus = post_dict["$beam_y-p"] # delta to off-momentum orbit

    def get_psv_trans(self):
        transverse = [[hit[var] for var in self.variables[:4]] for hit in self.bunch]
        mean_trans, cov_trans = xboa.common.fit_ellipse(transverse, self.eps_cut_trans)
        cov_trans /= numpy.linalg.det(cov_trans)**0.25 # determinant is 1
        self.cov_trans = cov_trans*self.emittance_trans*self.mass
        print(f"Found transverse covariance matrix\n{self.cov_trans}")
        psv_list = numpy.random.multivariate_normal([0, 0, 0, 0], self.cov_trans, self.n_particles)
        return psv_list

    def get_grid_long(self):
        n = self.n_particles
        mean_e = (self.p_mean**2+xboa.common.pdg_pid_to_mass[13]**2)**0.5
        longitudinal = [[0.0+(i-(n-1)/2)/self.frequency/(n-1), mean_e] for i in range(n)]
        return longitudinal

    def get_psv_long(self):
        longitudinal = [[hit[var] for var in self.variables[4:]] for hit in self.bunch if hit["weight"] > 0.5]
        mean_long, cov_long = xboa.common.fit_ellipse(longitudinal, self.eps_cut_long)
        print(f"Input mean:\n{mean_long}\ncov:\n{cov_long}")
        cov_long /= numpy.linalg.det(cov_long)**0.5 # determinant is 1
        cov_long *= self.emittance_long*self.mass/xboa.common.constants["c_light"]
        e_mean = (self.p_mean**2+self.mass**2)**0.5
        psv_list = numpy.random.multivariate_normal([0, e_mean], cov_long, self.n_particles)
        mean_long, cov_long = xboa.common.fit_ellipse(psv_list, self.eps_cut_long)
        print(f"Output mean:\n{mean_long}\noutput cov:\n{cov_long}")
        return psv_list

    def apply_dispersion(self, psv_all):
        disp_x = (self.beam_x_p_plus-self.beam_x_p_minus)/self.dispersion_dp
        disp_y = (self.beam_y_p_plus-self.beam_y_p_minus)/self.dispersion_dp
        print(f"Applying dispersion term using [dx, dy] = dpz * [{disp_x}, {disp_y}]")
        for i, psv in enumerate(psv_all):
            dpz = (psv[5]**2-(psv[1]**2+psv[3]**2+self.mass**2))**0.5-self.p_mean
            dx = dpz * disp_x
            dy = dpz * disp_y
            psv[0] += dx
            psv[2] += dy
            if False in numpy.isfinite(psv):
                print("Found nan in", i, dpz, psv)
        return psv_all

    def apply_a_pz_coefficient(self, psv_all):
        v_inv = numpy.linalg.inv(self.cov_trans)*self.emittance_trans
        print(f"Applying A-pz coefficient {self.a_pz_coefficient} on cov matrix\n{self.cov_trans}")
        for i, psv in enumerate(psv_all):
            psv_trans = psv[0:4]
            amplitude = numpy.dot(numpy.transpose(psv_trans), v_inv)
            amplitude = numpy.dot(amplitude, psv_trans)
            pz_in = psv[5]
            pz_out = pz_in + amplitude*self.a_pz_coefficient
            if i < 5:
                print(f"   {i:04d} A: {amplitude:8.4g} pz_in: {pz_in:8.4g} pz_out: {pz_out:8.4g} delta:{pz_out-pz_in:8.4g}")
            psv[5] = pz_out
        return psv_all


    def remove_nan(self, psv_all):
        remove_rows = []
        for row, psv in enumerate(psv_all):
            if False in numpy.isfinite(psv):
                remove_rows.append(row)
        print("Before cleanup shape was", psv_all.shape, "   ... Flagged rows", remove_rows, "for removal")
        psv_all = numpy.delete(psv_all, remove_rows, 0)
        print("After cleanup shape was", psv_all.shape)
        return psv_all

    def get_input_psv_list(self):
        bunch_list = xboa.bunch.Bunch.new_list_from_read_builtin(self.file_format, self.beam_file)
        print([bunch[0]["station"] for bunch in bunch_list])
        self.bunch = [bunch for bunch in bunch_list if bunch[0]["station"] == self.station][0]
        self.apply_cuts() # just a pid cut for now
        print("Got bunch for station", self.station, "with", self.bunch.bunch_weight(), "good hits")
        psv_transverse_list = self.get_psv_trans()
        if self.do_scan:
            psv_long_list = self.get_grid_long()
        else:
            psv_long_list = self.get_psv_long()
        psv_all = numpy.concatenate((psv_transverse_list, psv_long_list), 1)
        psv_all = self.apply_a_pz_coefficient(psv_all)
        psv_all = self.apply_dispersion(psv_all)
        psv_all = self.remove_nan(psv_all)
        psv_list = [dict([
                    (self.variables[col], psv_all[row][col]) for col in range(psv_all.shape[1])
                ]) for row in range(psv_all.shape[0])]
        print("Generated mean\n", numpy.mean(psv_all, axis=0))
        print("Generated covariance matrix\n", numpy.cov(psv_all, rowvar=False))
        return psv_list

    def get_bunch(self, station, hit_list_of_lists):
        bunch_hits = []
        for hit_list in hit_list_of_lists:
            for hit in hit_list:
                if hit["station"] == station and hit["pid"] == -13:
                    bunch_hits.append(hit)
        bunch = xboa.bunch.Bunch.new_from_hits(bunch_hits)
        return bunch

    def get_calc(self, station, hit_list_of_lists):
        bunch = self.get_bunch(station, hit_list_of_lists)
        score = {}
        print("Weight before cuts", bunch.bunch_weight())
        print("Weight after cut", end=" ")
        for i in range(5):
            eperp = bunch.get_emittance(["x", "y"])
            elong = bunch.get_emittance(["ct"])
            bunch.cut({"amplitude x y":eperp*4*5}, operator.gt, global_cut = False)
            bunch.cut({"amplitude ct":elong*2*5}, operator.gt, global_cut = False)
            print(bunch.bunch_weight(), end=" ")
        print()
        score["emit_perp"] = bunch.get_emittance(["x", "y"])
        score["emit_long"] = bunch.get_emittance(["ct"])
        score["emit_6d"] = bunch.get_emittance(["ct", "x", "y"])
        score["weight"] = bunch.bunch_weight()
        score["z"] = bunch[0]["z"]
        return score

    def get_score(self, hit_list_of_lists):
        # Score is sum upstream emittance ratios + transmission ratio. Note that we
        # cannot do a transmission cut in the experiment so we won't do it in
        # analysis
        score_us = self.get_calc(self.station_us, hit_list_of_lists)
        score_ds = self.get_calc(self.station_ds, hit_list_of_lists)
        transmission = score_ds["weight"]/score_us["weight"]
        emit_trans = score_ds["emit_perp"]/score_us["emit_perp"]
        emit_long = score_ds["emit_long"]/score_us["emit_long"]
        #score = emit_trans*emit_long/transmission/transmission
        score = emit_trans + emit_long + 1/transmission**2
        print("FigureOfMeritScore got score et:", emit_trans, "el:" emit_long, "trans:", transmission, "score:", score)
        # note low score is better - we are trying to minimise after all
        self.output = {
            "score_type":"CovScore",
            "us":score_us,
            "ds":score_ds,
            "total_score":score
        }
        return score

    def post_output(self): # optionally, we post output from the score function
        return {}

    def store_data(self):
        return self.output

    def test_calc(self, file_name):
        bunch_list = xboa.bunch.Bunch.new_list_from_read_builtin("icool_for009", file_name, "event_number")
        hit_list_of_lists = [bunch.hits() for bunch in bunch_list]
        print("Got score", self.get_score(hit_list_of_lists))
        print(self.store_data())
        return self.store_data()
