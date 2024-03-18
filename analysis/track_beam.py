import os
import json
import copy
import math
import shutil
import glob
import time

import numpy
import xboa.hit

#import optimisation_tools.plotting.plot_fields

from optimisation_tools.opal_tracking import OpalTracking
import optimisation_tools.utils.utilities as utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
from optimisation_tools.utils.twod_transfer_matrix import TwoDTransferMatrix
DecoupledTransferMatrix.det_tolerance = 1.

class BeamGen(object):
    def __init__(self, config):
        self.config = config

    def load_closed_orbit(self, closed_orbit_file = None):
        if not closed_orbit_file:
            closed_orbit_file = self.config.track_beam["closed_orbit_file"]
        file_name = os.path.join(self.config.run_control["output_dir"],
                                 closed_orbit_file)
        fin = open(file_name)
        self.closed_orbits  = json.loads(fin.read())

    def gen_beam(self):
        raise NotImplementedError("Not implemented")

    def update_subs(self, subs):
        # if required, update the substitutions
        return subs

    @classmethod
    def beam_setting(cls, beam, config):
        #if setting["beam"]["type"] == "last":
        #    station = setting["beam"]["station"]
        #    hit_list = [hit_list[station] for hit_list in self.last_tracking if station < len(hit_list)]
        if beam["type"] == "beam_gen":
            beam_gen = BeamShells(config, beam)
            hit_list = beam_gen.gen_beam()
        elif beam["type"] == "grid":
            beam_gen = BeamGrid(config, beam)
            hit_list = beam_gen.gen_beam()
        elif beam["type"] == "multibeam":
            beam_gen = MultiBeam(config, beam)
            hit_list = beam_gen.gen_beam()
        elif beam["type"] == "mvg":
            beam_gen = MVG(config, beam)
            hit_list = beam_gen.gen_beam()
        elif beam["type"] == "last" or beam["type"] == "recycle":
            beam_gen = Recycle(config, beam)
            hit_list = beam_gen.gen_beam()
        elif beam["type"] == "beam_file":
            beam_gen = BeamFile(config, beam)
            hit_list = beam_gen.gen_beam()
        else:
            raise ValueError("Did not recognise beam of type "+str(setting["beam"]["type"]))
        return hit_list

    MeV = 1.0e3
    GeV = 1.0


class MultiBeam(object):
    def __init__(self, config, beam):
        self.config = copy.deepcopy(config)
        self.beam = beam
        self.beam_list = []

    def gen_beam(self):    
        hit_list = []
        for beam in self.beam["beam_list"]:
            hit_list += BeamGen.beam_setting(beam, self.config)
        return hit_list


class MVG(BeamGen): # Multivariate Gaussian
    def __init__(self, config, beam):
        self.config = copy.deepcopy(config)
        self.beam = beam
        self.reference = beam["reference"]
        self.cov = None
        self.n_particles = beam["n_particles"]
        self.load_closed_orbit(beam["closed_orbit_file"])

    def get_cov(self, reference):
        """
            - D  = cov(q,E)*mean(E)/var(E)
            - D' = cov(p_q,E)*mean(E)/var(E)
        """
        elike = self.beam["variables_long"][-1]
        disp = [d*self.beam["cov_long"][1][1]/reference[elike] for d in self.beam["dispersion"]]
        tm = copy.deepcopy(self.closed_orbits[0]["tm"])
        for i, row in enumerate(tm):
            tm[i] = row[1:5]
        if self.config.track_beam["do_decoupling"]:
            self.tm = DecoupledTransferMatrix(tm, True)
        else:
            self.tm = TwoDTransferMatrix(tm, True)
        cov_trans = self.tm.get_v_m(self.beam["eigenemittances"])
        cov = [[x for x in row]+[0.0, disp[i]] for i, row in enumerate(cov_trans)]
        cov += [[0.0]*4 +self.beam["cov_long"][0]]
        cov += [disp+self.beam["cov_long"][1]]
        self.cov = numpy.array(cov)
        print(self.cov)

    def gen_beam(self):
        print("Tracking following particles:")
        var_list = self.beam["variables"]+self.beam["variables_long"]
        print(self.cov)
        pid = self.config.tracking["pdg_pid"]
        mass = xboa.common.pdg_pid_to_mass[abs(pid)]
        energy = self.beam["energy"]*self.MeV+mass
        pz = (energy**2-mass**2)**0.5
        ref_dict = {"energy":energy, "mass":mass, "pid":pid, "pz":pz}
        for i, var in enumerate(var_list):
            ref_dict[var] = self.reference[i]
        ref_hit = xboa.hit.Hit.new_from_dict(ref_dict, "pz")

        self.get_cov(ref_hit)
        point_list = numpy.random.multivariate_normal([0.0]*len(var_list), self.cov, self.n_particles)

        hit_list = [ref_hit]
        for point in point_list:
            print("Adding", point)
            hit_dict = {"energy":energy, "mass":mass, "pid":pid, "pz":pz}
            for i, var in enumerate(var_list):
                hit_dict[var] = self.reference[i]+point[i]
            hit = xboa.hit.Hit.new_from_dict(hit_dict, "pz")
            hit_list.append(hit)
        return hit_list

class BeamShells(BeamGen):
    def __init__(self, config, beam):
        self.config = copy.deepcopy(config)
        self.config.track_beam = beam
        self.tm = None
        self.mean = None
        self.betagamma = 1.0

    def get_mean(self, co):
        hit = xboa.hit.Hit.new_from_dict(co["seed_hit"])
        self.mean = [hit[var] for var in self.config.track_beam["variables"]]
        self.betagamma = hit["p"]/hit["mass"]

    def get_tm(self, co):
        tm = co["tm"]
        tm = copy.deepcopy(co["tm"])
        for i, row in enumerate(tm):
            tm[i] = row[1:5]
        if self.config.track_beam["do_decoupling"]:
            self.tm = DecoupledTransferMatrix(tm, True)
        else:
            self.tm = TwoDTransferMatrix(tm, True)

    def get_action(self, actions):
        dist = self.config.track_beam["amplitude_dist"]
        min_amp = 0.0
        max_amp = self.config.track_beam["max_amplitude_4d"]
        if dist == "grid":
            return actions[0], actions[1]
        au, av = -1, -1
        while au+av > max_amp or au+av < min_amp:
            if dist == "uniform":
                au = numpy.random.uniform(min_amp, actions[0])
                av = numpy.random.uniform(min_amp, actions[1])
            elif dist == "exponential":
                au = numpy.random.exponential(actions[0])
                av = numpy.random.exponential(actions[1])
            else:
                raise KeyError("Did not recognise amplitude_dist type "+str(dist))
        return au, av

    def get_phi(self, actions):
        n_phi_u = self.config.track_beam["n_per_dimension"]
        n_phi_v = self.config.track_beam["n_per_dimension"]
        if actions[0] == 0.0:
            n_phi_u = 1
        if actions[1] == 0.0:
            n_phi_v = 1
        phi_list = []
        dist = self.config.track_beam["phase_dist"]
        for ui in range(n_phi_u):
            for vi in range(n_phi_v):
                if dist == "grid":
                    phi_list.append([ui*2.*math.pi/n_phi_u-math.pi,
                                     vi*2.*math.pi/n_phi_v-math.pi])
                elif dist == "uniform":
                    phi_list.append([numpy.random.uniform(-math.pi, math.pi),
                                     numpy.random.uniform(-math.pi, math.pi)])
        return phi_list

    def get_aa_list(self, actions):
        aa_list = []
        phi_list = self.get_phi(actions)
        for phi_u, phi_v in phi_list:
            au, av = self.get_action(actions)
            aa = [phi_u, au, phi_v, av]
            aa_list.append(aa)
        return aa_list

    def get_psv_list(self, aa_list):
        psv_list = []
        for aa in aa_list:
            coupled = self.tm.action_angle_to_coupled(aa)
            psv = [coupled[i]+self.mean[i] for i in range(4)]
            psv_list.append(psv)
            print("aa:", aa, "coupled:", coupled, "psv:", psv)
        return psv_list

    def get_hit_list(self, trans_list, long_list):
        hit_list = []
        pid = self.config.tracking["pdg_pid"]
        for i, psv in enumerate(trans_list):
            t = long_list[i, 0]
            mass = xboa.common.pdg_pid_to_mass[abs(pid)]
            energy = long_list[i, 1]+mass
            pz = (energy**2-mass**2)**0.5
            hit_dict = {"energy":energy, "mass":mass, "pid":pid, "pz":pz, "t":t}
            for i, var in enumerate(self.config.track_beam["variables"]):
                hit_dict[var] = psv[i]
            hit = xboa.hit.Hit.new_from_dict(hit_dict, "pz")
            hit_list.append(hit)
        return hit_list

    def gen_longitudinal(self, n_events):
        dist = self.config.track_beam["longitudinal_dist"]
        if dist == "gaussian":
            return self.gen_gaussian_longitudinal(n_events)
        elif dist == "None" or not dist:
            return numpy.array([[0.0, self.config.track_beam["energy"]*self.MeV]]*n_events)
        else:
            raise KeyError("Did not recognise longitudinal_dist type "+str(dist))

    def gen_gaussian_longitudinal(self, n_events):
        mean = [0.0, self.config.track_beam["energy"]*self.MeV]
        cov = self.config.track_beam["longitudinal_params"]
        return numpy.random.multivariate_normal(mean, cov, n_events)


    def gen_beam(self):
        self.load_closed_orbit()
        hit_list = []
        for co in self.closed_orbits:
            self.get_mean(co)
            self.get_tm(co)
            eigen_emittance_list = self.config.track_beam["eigen_emittances"]
            for actions in eigen_emittance_list:
                aa_list = self.get_aa_list(actions) # action angle
                for aa in aa_list:
                    aa[1] /= self.betagamma
                    aa[3] /= self.betagamma
                trans_list = self.get_psv_list(aa_list) # phase space vector
                long_list = self.gen_longitudinal(len(trans_list))
                hit_list += self.get_hit_list(trans_list, long_list) # hit
        return hit_list

class BeamGrid(BeamGen):
    def __init__(self, config, beam):
        self.config = config
        self.beam = beam
        self.dim = len(beam["start"])
        self.reference = beam["reference"]
        self.start = beam["start"]
        self.stop = beam["stop"]
        self.nsteps = beam["nsteps"]
        self.step = [0.0 for i in range(self.dim)]
        self.hit_list = []
        for i in range(self.dim):
            if self.nsteps[i] == 1:
                continue
            self.step[i] = (beam["stop"][i]-beam["start"][i])/(beam["nsteps"][i]-1)
        
    def gen_grid(self):
        self.hit_list = []
        point = [0, 0, 0, 0, 0, 0]
        grid = [copy.deepcopy(point)]
        while True:
            for i in range(self.dim-1):
                if point[i] == self.nsteps[i]:
                    point[i] = 0
                    point[i+1] += 1
            if point[self.dim-1] == self.nsteps[self.dim-1]:
                break         
            grid.append(copy.deepcopy(point))
            point[0] += 1
        for point in grid:
            for i in range(self.dim):
                point[i] = self.start[i]+point[i]*self.step[i]
            self.hit_list.append(point)

    def make_hit(self, point):
        if "x'"  in self.beam["variables"] and\
           "y'" in self.beam["variables"]:
            return self.geometric_hit(point)
        elif "x'"  in self.beam["variables"] or\
             "y'" in self.beam["variables"]:
            raise KeyError("Cant mix geometric and normalised")
        else:
            return self.normalised_hit(point)

    def geometric_hit(self, point):
        pid = self.config.tracking["pdg_pid"]
        mass = xboa.common.pdg_pid_to_mass[abs(pid)]
        energy = self.beam["energy"]*self.MeV+mass
        hit_dict = {"energy":energy, "mass":mass, "pid":pid}
        hit_dict["pz"] = (energy**2-mass**2)**0.5/(1+point[1]**2+point[3]**2)**0.5
        for i, var in enumerate(self.beam["variables"]):
            hit_dict[var] = point[i]
        hit = xboa.hit.Hit.new_from_dict(hit_dict, "pz")
        return hit

    def normalised_hit(self, point):
        pid = self.config.tracking["pdg_pid"]
        mass = xboa.common.pdg_pid_to_mass[abs(pid)]
        energy = self.beam["energy"]*self.MeV+mass
        hit_dict = {"energy":energy, "mass":mass, "pid":pid}
        for i, var in enumerate(self.beam["variables"]):
            hit_dict[var] = point[i]
        hit = xboa.hit.Hit.new_from_dict(hit_dict, "pz")
        return hit

    def gen_hits(self):
        print("Tracking following particles:")
        hit_list = []
        for point in self.hit_list:
            print("Adding", point)
            hit_list.append(self.make_hit(point))
        if self.reference:
            ref_hit = self.make_hit(self.reference)
            hit_list = [ref_hit]+hit_list
        self.hit_list = hit_list

    def gen_beam(self):
        self.gen_grid()
        self.gen_hits()
        return self.hit_list

class Recycle(BeamGen):
    def __init__(self, config, beam):
        self.config = config
        self.beam = beam

    def gen_beam(self):
        station = self.beam["station"]
        if self.beam["type"] == "last":
            tracking = self.last_tracking
        elif self.beam["type"] == "recycle":
            setting_name = self.beam["setting_name"]
            tracking = self.tracking_store[setting_name]
        hit_list = [hit_list[station] for hit_list in self.last_tracking if station < len(hit_list)]
        hit_list = self.offset(hit_list)
        return hit_list
 
    def offset(self, hit_list):
        if "offset" not in self.beam:
            return hit_list
        offset = self.beam["offset"]
        print("Offsetting", offset)
        for hit in hit_list:
            for key, value in offset.items():
                hit[key] += value 
        return hit_list

    @classmethod
    def store_tracking(cls, name, tracks):
        cls.tracking_store[name] = tracks
        cls.last_tracking = tracks

    tracking_store = {}
    last_tracking = None


class BeamFile(BeamGen):
    def __init__(self, config, beam):
        self.config = config
        self.beam = beam

    def gen_beam(self):
        fname = self.beam["filename"]
        file_format = self.beam["format"]
        beam = xboa.bunch.Bunch.new_from_read_builtin(file_format, filename)
        hit_list = [hit for hit in beam]
        hit_list = self.offset(hit_list)
        return hit_list

    def offset(self, hit_list):
        if "offset" not in self.beam:
            return hit_list
        offset = self.beam["offset"]
        print("Offsetting", offset)
        for hit in hit_list:
            for key, value in offset.items():
                hit[key] += value
        return hit_list

    @classmethod
    def store_tracking(cls, name, tracks):
        cls.tracking_store[name] = tracks
        cls.last_tracking = tracks

    tracking_store = {}
    last_tracking = None



class TrackBeam(object):
    def __init__(self, config):
        self.config = config
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.track_beam["run_dir"])
        self.last_tracking = None
        self.tracking_store = {}

    def setup(self):
        #self.beam_gen = BeamShells(self.config)
        self.hit_list = None #self.beam_gen.gen_beam()

    def save_tracking(self, out_dir):
        src_dir = self.tmp_dir
        base_dir = os.path.join(self.config.run_control["output_dir"],
                                self.config.track_beam["save_dir"])
        target_dir = os.path.join(base_dir, out_dir)
        print("Saving to", target_dir)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(src_dir, target_dir)
        time.sleep(1)

    def do_tracking(self):
        if len(self.config.substitution_list) > 1:
            raise RuntimeError("Didnt code subs list > 1")
        utilities.clear_dir(self.tmp_dir)
        here = os.getcwd()
        os.chdir(self.tmp_dir)

        #self.config.tracking["analysis_coordinate_system"] = "opal"
        for setting in self.config.track_beam["settings"]:
            self.track_setting(setting)
        print("done")
        os.chdir(here)

    def beam_setting(self, setting):
        self.hit_list = BeamGen.beam_setting(setting["beam"], self.config)

    def track_setting(self, setting): 
        # somewhere in here there is a bug - subs list is getting overwritten
        print("Starting setting", setting["name"], "#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#")      
        self.tracking = utilities.setup_tracking(self.config,
                                                  setting["probe_files"],
                                                  setting["beam"]["energy"]) # energy is in GeV
        self.beam_setting(setting)

        self.config.substitution_list[0].update({"__n_particles__":len(self.hit_list)})
        utilities.do_lattice(self.config,
                             self.config.substitution_list[0],
                             setting["subs_overrides"], tracking = self.tracking)
        if setting["direction"] == "backwards":
            for hit in self.hit_list:
                for var in "px", "py", "pz":
                    hit[var] *= -1
                pid = self.config.tracking["pdg_pid"]
                mass = xboa.common.pdg_pid_to_mass[abs(pid)]
                hit["z"] = 0.0
        elif setting["direction"] != "forwards":
            raise RuntimeError("Direction must be forwards or backwards")

        print("    ... tracking", len(self.hit_list), "hits", setting["direction"])
        if self.config.track_beam["print_events"] == "all":
            print_events = [i for i in range(len(self.hit_list))]
        else:
            print_events = self.config.track_beam["print_events"]
        for i in print_events:
            print("      Event", i, end="  ")
            for var in ["x", "y", "z", "x'", "y'", "kinetic_energy", "t"]:
                print(var+":", self.hit_list[i][var], end=" ")
            print()
        try:
            self.last_tracking = self.tracking.track_many(self.hit_list)
        except (IOError, OSError): # maybe we chose not to open any files
            print("Caught error while attempting to open files '{0}'".format(setting["probe_files"]))
            if setting["probe_files"]:
                raise
            else:
                print("Ignoring - probe files was false")
                self.last_tracking = []
        Recycle.store_tracking(setting["name"], self.last_tracking)


        print("    ... found", [len(hit_list) for hit_list in self.last_tracking], "output hits")
        print("    from", self.tracking.get_name_dict())

        self.save_tracking(setting["name"])

        for filename in glob.glob("*.h5"):
            os.unlink(filename)
        print()

def main(config):
    tracker = TrackBeam(config)
    tracker.setup()
    tracker.do_tracking()
