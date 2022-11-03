import json
import sys
import math
import copy
import os
import shutil
import time

import ROOT
import numpy
numpy.seterr('warn')
numpy.set_printoptions(linewidth=200)

from xboa.hit import Hit
import xboa.common

#from PyOpal.polynomial_coefficient import PolynomialCoefficient
#from PyOpal.polynomial_map import PolynomialMap
from optimisation_tools.utils.polynomial_fitter import PolynomialFitter
import optimisation_tools.utils.utilities as utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix

#Jobs:
#* rotating planes might work; now need to add option to tracking to do analysis in reference coordinate system
#* fix RK4 - comparison is pretty good between RK4 and tracking, but not perfect. Check!

class ClosedOrbitFinder4D(object):
    def __init__(self, config):
        self.config = config
        self.config_co = config.find_closed_orbits
        self.energy = self.config.substitution_list[0]["__energy__"]
        self.run_dir = os.path.join(self.config.run_control["output_dir"],
                                    self.config_co["run_dir"])
        self.centroid = None
        self.var_list_1 = self.config_co["vars"]
        #self.var_list_2 = ["x", "px", "y", "py", "t", "kinetic_energy"]
        self.var_list = self.var_list_1
        self.subs_tracking = {}
        self.output_list = []
        self.print_list = []
        self.py_opal_setup = False
        self.minuit_score_list = []
        self.get_tracking(True)

    def get_tracking(self, clear):
        self.here = os.getcwd()
        probes = self.config_co["probe_files"]
        if clear:
            utilities.clear_dir(self.run_dir)
        self.tracking = utilities.setup_tracking(self.config, probes, self.energy)

    def seed_to_hit(self, seed, t):
        hit = utilities.reference(self.config, self.energy)
        hit["t"] = t
        for i, var in enumerate(self.var_list):
            hit[var] = seed[i]
        hit.mass_shell_condition("pz") # adjust pz so E^2 = p^2 + m^2
        return hit

    @classmethod
    def rotation_matrix(cls, r1, r2):
        # rotation matrix that rotates r1 onto r2
        # note this is not uniquely defined
        v = numpy.cross(r1, r2)/numpy.linalg.norm(r1)/numpy.linalg.norm(r2)
        st = numpy.linalg.norm(v)
        v /= st
        ct = math.cos(math.asin(st))
        rotation_matrix = [
            [v[0]*v[0]*(1-ct)+ct,      -v[2]*st+v[0]*v[1]*(1-ct), v[0]*v[2]*(1-ct)+v[1]*st],
            [v[0]*v[1]*(1-ct)+v[2]*st,       ct+v[1]*v[1]*(1-ct), v[1]*v[2]*(1-ct)-v[0]*st],
            [v[0]*v[2]*(1-ct)-v[1]*st,  v[0]*st+v[1]*v[2]*(1-ct), v[2]*v[2]*(1-ct)+ct],
        ]
        return numpy.array(rotation_matrix)

    def get_vector(self, hit_list):
        tm_list = []
        for hit in hit_list:
            vector = [hit[var] for var in self.var_list]
            tm_list.append(vector)
        return tm_list

    def track_many(self, seed_list, t, final_subs):
        overrides = self.config_co["subs_overrides"]
        if final_subs:
            overrides = self.config_co[final_subs]
        overrides["__n_particles__"] = len(seed_list)+1
        hit_list = []
        os.chdir(self.run_dir)
        self.subs_tracking = utilities.do_lattice(self.config, self.subs, overrides)
        if self.config_co["use_py_tracking"]:
            self.tracking = utilities.setup_py_tracking(self.config, self.run_dir, self.config_co["py_tracking_phi_list"])
        for seed in seed_list:
            hit = self.seed_to_hit(seed, t)
            hit_list.append(hit)
        track_list = []
        if len(hit_list) > 0:
            hit_list.insert(0, hit_list[0])
            track_list = self.tracking.track_many(hit_list)
        os.chdir(self.here)
        return track_list

    def get_decoupled(self, tm):
        m = tm
        print("get decoupled - before cut")
        for row in m:
            for element in row:
                print(format(element, "8.4g"), end=" ")
            print()
        dim = len(m)
        m = [row[1:5] for row in m[0:4]]
        print("get decoupled - after cut")
        for row in m:
            print(row)
        print("done")
        DecoupledTransferMatrix.det_tolerance = 1e9
        decoupled = DecoupledTransferMatrix(m)
        return decoupled

    def get_co(self, tm):
        # \vec{x} = \vec{m} + \matrix{M} \vec{x}
        # Solve (\matrix{1} - \matrix{M}) \vec{x} = \vec{m}
        m = tm
        print("TM:")
        for i, row in enumerate(m):
            m[i] = row[0:5]
        #print(self.str_matrix(m, fmt="20.6f"))
        try:
            dm = self.get_decoupled(tm)
            print(self.str_matrix(m, fmt="20.6f"))
            print("Determinant:  ", numpy.linalg.det(dm.m))
            print("Symplecticity:")
            print(self.str_matrix(dm.simplecticity(dm.m), fmt="20.6f"))
            print("Tune:", [dm.get_phase_advance(i)/math.pi/2. for i in range(2)])
        except Exception:
            sys.excepthook(*sys.exc_info())
            sf = math.sin(math.pi/4)
            cf = math.cos(math.pi/4)
            dm = [[cf, sf, 0, 0], [-sf, cf, 0, 0], [0, 0, cf, -sf], [0, 0, sf, cf]]
            print("Failed to calculate phase advance")
        dim = len(m)
        m_vec = numpy.array([m[i][0] for i in range(dim)])
        matrix = numpy.array([[-m[i][j] for j in range(1, dim+1)] for i in range(dim)])
        for i in range(dim):
            matrix[i, i] += 1.
        inv_matrix = numpy.linalg.inv(matrix)
        x_vec = numpy.dot(inv_matrix, m_vec.transpose())
        m_orig = numpy.array([[m[i][j] for j in range(1, dim+1)] for i in range(dim)])
        x_vec = [x_vec[i] for i in range(dim)]
        return x_vec, dm

    def str_matrix(self, matrix, fmt="10.6g"):
        output_string = ""
        for item in matrix:
            try:
               output_string += self.str_matrix(item, fmt)+"\n"
            except TypeError:
               output_string += format(item, fmt)+" "
        return output_string

    def get_values(self, dim): # makes a hyperdiamond at +- 1
        value_list = [[0. for i in range(dim)]]
        for d in range(dim):
            for delta in [-1, 1]:
                value = [0. for i in range(dim)]
                value[d] = delta
                value_list.append(value)
        for value in value_list:
            yield value

    def fit_matrix(self, tm_list_in, tm_list_out):
        coefficients = []
        for i in range(50):
            index = PolynomialMap.index_by_power(i, 4)
            do_continue = False
            do_break = False
            sum_index = []
            for i, n in enumerate(index):
                if n > 1:
                    do_break = True
                sum_index += [i]*n
            if len(sum_index) < 2:
                do_continue = True
            if do_break:
                break
            if do_continue:
                continue
            coefficients += [PolynomialCoefficient(sum_index, j, 0.0) for j in range(4)]
        tm = PolynomialMap.least_squares(tm_list_in, tm_list_out, 1, coefficients)
        return tm.get_coefficients_as_matrix()

    def fit_matrix_2(self, tm_list_in, tm_list_out):
        tm_list_in = numpy.array(tm_list_in)
        tm_list_out = numpy.array(tm_list_out)
        fitter = PolynomialFitter(4)
        tm = fitter.fit_transfer_map(tm_list_in, tm_list_out)
        #fitter.print_array(tm)
        return tm


    def get_tm(self, seeds, deltas, cell_list, final_subs=False):
        dim = len(seeds)
        tm_list_of_lists = []
        value_list = [copy.deepcopy(seeds)]
        for j, values in enumerate(self.get_values(dim)):
            value_list.append([seeds[i]+values[i]*deltas[i] for i in range(dim)])
        track_list = self.track_many(value_list, 0., final_subs)[1:]
        if cell_list == None:
            n_hits = min([len(track) for track in track_list])
            cell_list = range(n_hits)
        try:
            for cell in cell_list:
                tm_list = self.get_vector([a_track[cell] for a_track in track_list])
                print(self.str_matrix(tm_list, "14.8g"))
                tm_list_of_lists.append(tm_list)
        except IndexError:
            err = "Output had "+str(len(track_list))+" tracks with"
            err += str([len(track) for track in track_list])+" track points. "
            err += "Require "+str( max(cell_list) )+" track points."
            print(err)
            raise RuntimeError(err) from None
        return tm_list_of_lists, track_list[0]

    def get_error(self, delta):
        scale = self.config_co["deltas"]
        error = sum([(delta[i]/scale[i])**2 for i in range(len(delta))])
        error = error**0.5
        return error

    def print_ref_track(self, ref_track, seeds, tm):
        print("Reference track for seed", seeds)
        for i, hit in enumerate(ref_track):
            ref_list = [hit[var] for var in self.var_list+["t", "kinetic_energy"]]
            fmt = "10.6g"
            if not tm:
                fmt = "14.10g"
            print("Ref track:", self.str_matrix(ref_list, fmt), end=' ')
            if not tm:
                print()
                continue
            print("decoupled", end=' ')
            coupled = [hit[var]-seeds[j] for j, var in enumerate(self.var_list)]
            #print coupled
            try:
                decoupled = tm.decoupled(coupled)
                print(self.str_matrix(decoupled))
            except Exception:
                continue

    def tm_co_fitter(self, seeds, max_iter = None):
        output = {}
        dim = len(seeds)
        print("DIMENSION", dim)
        tolerance = self.config_co["tolerance"]
        if max_iter == None:
            max_iter = self.config_co["max_iterations"]
        if max_iter == 0:
            return {
                "seed":seeds,
                "tm":[[0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.]],
                "substitutions":utilities.do_lattice(self.config,
                                                           self.subs,
                                                           self.config_co["subs_overrides"]),
                "errors":[-1],
            }
        deltas = self.config_co["deltas"][0:len(self.var_list)]
        a_track = None
        tm = None
        new_deltas = copy.deepcopy(deltas)
        errors = []
        new_seeds = seeds
        while len(errors) == 0 or (errors[-1] > tolerance and len(errors) < max_iter):
            print("----------------\nLooping with seed", self.str_matrix(new_seeds), end=' ')
            print("delta", self.str_matrix(deltas, "4.2g"), end=' ')
            print("error", self.get_error(new_deltas), "tolerance", tolerance)
            self.centroid = self.seed_to_hit(new_seeds, 0.)
            try:
                print("tracking in")
                cell_list = [self.config_co["us_cell"], self.config_co["ds_cell"]]
                tm_list_of_lists, ref_track = self.get_tm(new_seeds, deltas, cell_list)
                tm_list_in = tm_list_of_lists[0]
                tm_list_out = tm_list_of_lists[1]
                print("tracking out")
            except Exception:
                # if tm calculation fails, we abort with the last successful result
                # stored in seeds (e.g. tracking not convergent)
                print("Tracking found the following files:", self.tracking.name_dict)
                sys.excepthook(*sys.exc_info())
                break
            seeds = new_seeds
            tm = self.fit_matrix_2(tm_list_in, tm_list_out)
            new_co, dm = self.get_co(tm)
            print("New closed orbit", new_co)
            self.print_ref_track(ref_track, seeds, dm)
            for i in range(dim):
                new_deltas[i] = abs(tm_list_out[0][i] - tm_list_in[0][i])
                if self.config_co["adapt_deltas"] and new_deltas[i] < deltas[i]:
                    deltas[i] = new_deltas[i]
            errors.append(self.get_error(new_deltas))
            #new_seeds = [seeds[i]+x for i, x in enumerate(new_co)]
            new_seeds = [x for i, x in enumerate(new_co)]
        print("Finished iteration with deltas", deltas, "rms", sum([d*d for d in deltas])**0.5)
        output = {
            "seed":seeds,
            "tm":tm,
            "substitutions":self.subs_tracking,
            "errors":errors,
        }
        return output

    def get_new_seed(self, config_seed):
        if len(self.output_list) == 0 or len(self.output_list[-1]) == 0:
            return config_seed
        elif len(self.output_list[-1]) == 1:
            seed = Hit.new_from_dict(self.output_list[-1][-1]["seed_hit"])
            return [seed[var] for var in self.var_list]
        else:
            seed0 = Hit.new_from_dict(self.output_list[-1][-2]["seed_hit"])
            seed1 = Hit.new_from_dict(self.output_list[-1][-1]["seed_hit"])
            s0 = [seed0[var] for var in self.var_list]
            s1 = [seed1[var] for var in self.var_list]
            s2 = [2*s1[i]-s0[i] for i in range(4)]
            return s2

    def save_track_orbit(self):
        file_name = self.config_co["orbit_file"]
        if file_name == "" or type(file_name) != type(""):
            return
        first, last = file_name.split(".")
        file_out = first+"_"+str(self.run_index)+"."+last
        os.rename(self.run_dir+"/"+file_name, self.run_dir+"/"+file_out)
        self.run_index += 1

    def get_subs_string(self):
        subs_dict = [{"subs":sub} for sub in self.config.substitution_list]
        subs_axes = utilities.get_substitutions_axis(subs_dict, "subs")
        self.print_list = []
        for i, subs in enumerate(self.config.substitution_list):
            print_str = ""
            for axis in subs_axes:
                print_str += "    "+utilities.sub_to_name(axis)
                print_str += utilities.sub_to_units(axis)+": "
                print_str += str(subs_axes[axis][i])+"\n"
            self.print_list.append(print_str)

    def find_closed_orbits(self):
        self.output_list = []
        self.get_subs_string()
        for config_seed in self.config_co["seed"]:
            self.output_list.append([])
            for i, self.subs in enumerate(self.config.substitution_list):
                print("\n\nNew closed orbit loop", i+1, "/", len(self.config.substitution_list), "with lattice values")
                print(self.print_list[i])
                self.energy = self.subs["__energy__"]
                self.get_tracking(False)
                seed = self.get_new_seed(config_seed)
                self.centroid = self.seed_to_hit(seed, 0.)
                a_track = None
                try:
                    self.var_list = self.var_list_1
                    output = self.tm_co_fitter(seed)
                    output["seed_in"] = seed
                    output["seed_hit"] = self.seed_to_hit(output["seed"], 0.).dict_from_hit()
                except Exception:
                    sys.excepthook(*sys.exc_info())
                if self.config_co["do_minuit"]:
                    print("DOING MINUIT WITH seed", output["seed"])
                    minuit_out = self.find_co_minuit(output["seed"])
                    output["seed"] = minuit_out
                    output["seed_hit"] = self.seed_to_hit(output["seed"], 0.).dict_from_hit()
                    try:
                        print("DOING post seed with ", minuit_out)
                        output = self.tm_co_fitter(minuit_out, 1)
                        output["seed_hit"] = self.seed_to_hit(minuit_out, 0.).dict_from_hit()
                    except Exception:
                        print("\n*****************")
                        print("Attempting to find the tune after minuit... raised an exception; lattice may be unstable.")
                        print("*****************\n")
                        sys.excepthook(*sys.exc_info())
                try:
                    self.output_list[-1].append(output)
                    deltas = self.config_co["deltas"][0:len(self.var_list)]
                    tm_list_of_lists, a_track = self.get_tm(output["seed"], deltas, None, "final_subs_overrides")
                    output["ref_track"] = [hit.dict_from_hit() for hit in a_track]
                    output["tm_list"] = [self.fit_matrix_2(tm_list_of_lists[0], tm_list) for tm_list in tm_list_of_lists]
                    self.print_ref_track(a_track, output["seed"], None)
                    self.track_many([], 0.0, "plotting_subs")
                except Exception:
                    sys.excepthook(*sys.exc_info())
                self.save_track_orbit()
                self.save_output(self.output_list, False)
        self.save_output(self.output_list, self.config_co["overwrite"])

    def save_output(self, output_list, overwrite):
        print("Overwriting closed orbits")
        output_dir = self.config.run_control["output_dir"]
        tmp = output_dir+"/"+self.config_co["output_file"]+"_tmp"
        fout = open(tmp, "w")
        for output in output_list:
            print(json.dumps(output), file=fout)
        print("Saved to", tmp)
        fout.close()
        if overwrite:
            output = output_dir+"/"+self.config_co["output_file"]
            os.rename(tmp, output)

    def find_co_minuit(self, seed):
        print("Running minuit", seed)
        seed_hit = self.seed_to_hit(seed, 0)
        force = self.config_co["fixed_variables"]
        n_iterations = self.config_co["minuit_iterations"]
        target_score = self.config_co["minuit_tolerance"]
        self.opt_errs = self.config_co["minuit_weighting"]
        deltas = self.config_co["deltas"]
        self.iteration_number = 1
        self.minuit = ROOT.TMinuit(len(self.var_list))
        self.minuit_stations = [self.config_co["us_cell"], self.config_co["ds_cell"]]
        for i, var in enumerate(self.var_list):
            if var in force:
                seed_hit[var] = force[var]
            self.minuit.DefineParameter(i, var, seed_hit[var], deltas[i], 0, 0)
            if var in force:
                self.minuit.FixParameter(i)

        self.minuit.SetFCN(self.minuit_function)
        try:
            self.minuit.Command("SIMPLEX "+str(n_iterations)+" "+str(target_score))
        except Exception:
            print(f"Terminated after {self.iteration_number}/{n_iterations} iterations")
        return self.get_minuit_hit()

    def get_minuit_hit(self):
        seed = [None]*4
        for i, var in enumerate(self.var_list):
            x = ROOT.Double()
            err = ROOT.Double()
            self.minuit.GetParameter(i, x, err)
            seed[i]  = float(x)
        hit = self.seed_to_hit(seed, 0)
        return seed


    def minuit_function(self, nvar, parameters, score, jacobian, err):
        seed_in = self.get_minuit_hit()
        hit_list = self.track_many([seed_in]*3, 0.0, False)[1]
        print("Iteration", self.iteration_number, "with seed", seed_in)
        for i, hit in enumerate(hit_list):
            print(format(i, "4d"), end="")
            if i in self.minuit_stations:
                print("*", end=" ")
            else:
                print(" ", end=" ")
            for var in self.var_list:
                print(var.ljust(4), format(hit[var], "14.10g"), end=" ")
            print()
        score[0] = 0.0
        print("Scores")
        for var in self.var_list:
            try:
                value_list = [hit_list[i][var] for i in self.minuit_stations]
            except IndexError:
                if self.iteration_number == 0:
                    print("Seed did not work")
                    raise
                else:
                    score[0] = max(self.minuit_score_list)*10
            delta = numpy.std(value_list)
            score[0] += delta/self.opt_errs[var]
            print(var.ljust(4), format(delta, "14.10g"), end=" ")
        print("Total", score[0])
        self.minuit_score_list.append(score[0])
        self.iteration_number += 1
        if self.iteration_number > self.config_co["minuit_iterations"]:
            raise StopIteration("Max iterations exceeded")

    run_index = 1

def main(config):
    co_finder = ClosedOrbitFinder4D(config)
    co_finder.find_closed_orbits()

if __name__ == "__main__":
    main()
    if len(sys.argv) == 1:
        input()

