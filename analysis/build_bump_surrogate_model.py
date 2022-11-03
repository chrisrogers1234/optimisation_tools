"""
Script to find the RF set up; drives find closed orbit
"""

import os
import sys
import copy
import json
import math

import numpy
import matplotlib

import xboa.common
from xboa.hit import Hit
sys.path.insert(1, "scripts")
from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix
import optimisation_tools.analysis.bump_surrogate_model.minimiser as minimiser
#from PyOpal import polynomial_patch
#from PyOpal import ndgrid

class BuildBumpSurrogateModel(object):
    def __init__(self, config):
        self.config = config
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.build_bump_surrogate_model["run_dir"])
        self.field_names = [name for name in self.field_names_gen()]
        self.field_names = sorted(self.field_names,
                        key = lambda key: [x for x in reversed(key.split('_'))])
        self.dummy_run = False
        self.subs = {}
        self.tracking_result = []
        self.grid_list = []
        self.seed_list = []
        self.map_list = []
        self.inverse_map_list = []
        self.value_lists = []
        self.optimisation = None
        self.f_size = 18
        DecoupledTransferMatrix.det_tolerance = 1.

    def field_names_gen(self):
        n_h_bumps = self.config.build_bump_surrogate_model["n_h_bumps"]
        n_v_bumps = self.config.build_bump_surrogate_model["n_v_bumps"]
        for i in range(n_h_bumps):
            yield "__h_bump_"+str(i+1)+"_field__"
        for i in range(n_v_bumps):
            yield "__v_bump_"+str(i+1)+"_field__"

    def store_data(self):
        outname = self.get_filename_root()+"_"+f'{self.store_index:03}'+".out"
        tmpname = self.get_filename_root()+".tmp"
        print("Moving data from", tmpname, "to", outname)
        os.rename(tmpname, outname)
        self.store_index += 1

    def track_map(self):
        if not self.config.build_bump_surrogate_model["track_map"]:
            return
        try:
            os.rename(
                self.get_filename_root()+".tmp",
                self.get_filename_root()+".old"
            )
        except OSError:
            pass
        for i, subs in enumerate(self.config.substitution_list):
            self.subs = subs
            for opt_i, optimisation in enumerate(self.config.build_bump_surrogate_model["staged_optimisation"]):
                optimisation["index"] = opt_i
                self.optimisation = optimisation
                self.build_mesh()
                print("Tracking optimisation", opt_i)
                self.track_mesh()
        os.rename(self.get_filename_root()+".tmp", self.get_filename_root()+".out")

    def get_hit(self, tracking):
        foil_station = self.config.build_bump_surrogate_model["bump_probe_station"]
        for hit in tracking:
            if hit[0] == foil_station:
                return hit[1:]

    def build_one_map(self, value_list):
        fit_order = self.config.build_bump_surrogate_model["fit_order"]
        smoothing_order = self.config.build_bump_surrogate_model["smoothing_order"]
        grid = self.build_ndgrid()
        my_map = polynomial_patch.PolynomialPatch.initialise_from_solve_factory(
                                                        grid,
                                                        value_list,
                                                        fit_order,
                                                        smoothing_order)
        print("Build one map")
        self.build_mesh()
        for i, point in enumerate(self.mesh):
            test_value = my_map.function(point)
            diff = [format(test_value[j]-value_list[i][j], "4.8g") for j in range(4)]
            print("   ", point, value_list[i], test_value, diff)

        self.grid_list.append(grid)
        self.value_lists.append(value_list)
        self.map_list.append(my_map)

    def build_map(self):
        if not self.config.build_bump_surrogate_model["build_map"]:
            return
        fin = open(self.get_filename_root()+".out")
        field = None
        opt_i = None
        value_list = []
        for line in fin.readlines():
            mesh_point_data = json.loads(line)
            self.optimisation = mesh_point_data["optimisation"]
            if self.optimisation["index"] != opt_i:
                fields = self.get_fields()
                if opt_i != None:
                    self.build_one_map(value_list)
                    value_list = []
                opt_i = self.optimisation["index"]
            point = [mesh_point_data["subs"][field] for field in fields]
            value = self.get_hit(mesh_point_data["tracking"])
            if value == None:
                print("No value - skipping point", point)
                continue
            value_list.append(value)
        self.build_one_map(value_list)

    def get_fields(self):
        field_names = copy.deepcopy(self.field_names)
        for name in self.optimisation["fix_bumps"]:
            try:
                index = field_names.index(name)
                del field_names[index]
            except ValueError:
                pass
        return field_names

    def build_ndgrid(self):
        field_names = self.get_fields()
        n_dimensions = len(field_names)
        n_per_dimension = self.optimisation["n_per_dimension"]
        field_scale = self.optimisation["field_scale"]
        n_points = [n_per_dimension]*n_dimensions
        start = [-field_scale]*n_dimensions
        step = [field_scale*2/(n_per_dimension-1)]*n_dimensions
        my_grid = ndgrid.NDGrid.initialise_fixed_spacing(n_points, step, start)
        return my_grid

    def build_mesh(self, n_per_dimension = None):
        field_names = self.get_fields()
        n_dimensions = len(field_names)
        if n_per_dimension == None:
            n_per_dimension = self.optimisation["n_per_dimension"]
        field_scale = self.optimisation["field_scale"]
        mesh = xboa.common.make_grid(n_dimensions, n_per_dimension)
        if n_per_dimension == 1:
            mesh = [numpy.matrix([[0.0 for i in range(n_dimensions)]])]
        for point in mesh:
            point *= field_scale
        mesh = [item.tolist()[0] for item in mesh]
        self.mesh = sorted(mesh)

    def get_filename_root(self):
        fname = self.config.run_control["output_dir"]+"/"
        fname += self.config.build_bump_surrogate_model["output_file"]
        return fname

    def save_state(self, suffix, subs_overrides):
        state = {
            "tracking":copy.deepcopy(self.tracking_result),
            "subs":subs_overrides,
            "optimisation":self.optimisation,
        }
        fname = self.get_filename_root()+"."+suffix
        fout = open(fname, "a")
        print(json.dumps(state), file=fout)

    def track_mesh(self):
        mesh_field_names = self.get_fields()
        subs_overrides = self.config.build_bump_surrogate_model["subs_overrides"]
        subs_overrides["__beam_phi_init__"] = self.optimisation["phi_start"]
        for i, field_setting in enumerate(self.mesh):
            for f_name in self.field_names:
                subs_overrides[f_name] = 0.
                if f_name in mesh_field_names:
                    f_index = mesh_field_names.index(f_name)
                    subs_overrides[f_name] = field_setting[f_index]
                subs_overrides[f_name] += self.optimisation["seed_fields"][f_name]
            print("\r", i, "/", len(self.mesh), end="")
            sys.stdout.flush()
            self.track_one(subs_overrides)
            self.save_state("tmp", subs_overrides)
        print()


    def track_one(self, subs_overrides):
        utilities.clear_dir(self.tmp_dir)
        os.chdir(self.tmp_dir)
        probes = self.config.build_bump_surrogate_model["ref_probe_files"]
        ref_energy = self.config.build_bump_surrogate_model["energy"]
        tracking = utilities.setup_tracking(self.config, probes, ref_energy)
        test_hit = tracking.ref.deepcopy()
        closed_orbit = self.config.build_bump_surrogate_model["closed_orbit"]
        test_hit["x"] = closed_orbit[0]
        test_hit["px"] = closed_orbit[1]
        test_hit["y"] = closed_orbit[2]
        test_hit["py"] = closed_orbit[3]
        # fix momentum
        test_hit["pz"] = (tracking.ref["p"]**2-test_hit["px"]**2)**0.5
        #print("Reference kinetic energy:", tracking.ref["kinetic_energy"])
        #print("Seed kinetic energy:     ", test_hit["kinetic_energy"], flush=True)
        if not self.optimisation["fore_tracking"]:
            test_hit["px"] *= -1
            test_hit["py"] *= -1
            test_hit["pz"] *= -1
            subs_overrides["__beam_charge__"] = -1
        utilities.do_lattice(self.config, self.subs, subs_overrides)           
        if self.dummy_run: # dummy run
            fields = [2*subs_overrides[fname] for fname in self.get_fields()]
            self.tracking_result = [[i]+fields for i in range(10)]
        else:
            hit_list = tracking.track_many([test_hit])[1]
            self.tracking_result = [[hit["station"], hit["x"], hit["px"], hit["y"], hit["py"]] for hit in hit_list]
            #print("Station to probe mapping:\n   ", end=' ')
            #for fname, i in tracking.get_name_dict().items():
            #    print("("+str(i)+",", fname+")", end=' ')
            #print()

    def build_inverse_map(self):
        self.inverse_map_list = []
        self.seed_list = []
        for i in range(len(self.grid_list)):
            print("Mapping", i)
            self.optimisation = self.config.build_bump_surrogate_model["staged_optimisation"][i]
            self.build_mesh( self.optimisation["inverse_n_per_dimension"])
            self.seed_list.append(sorted(self.mesh))
            print("    with seed list length", len(self.seed_list[-1]))
            self.inverse_map_list.append(self.inverse_mapping(i))

    def get_limits(self):
        limits = []
        for i in range(4):
            limits.append([-self.optimisation["field_scale"]*101,
                            self.optimisation["field_scale"]*101])
        return limits

    def inverse_mapping(self, opt):
        return None
        fit_order = self.config.build_bump_surrogate_model["fit_order"]
        smoothing_order = self.config.build_bump_surrogate_model["smoothing_order"]
        self.optimisation = self.config.build_bump_surrogate_model["staged_optimisation"][opt]

        seed_list = self.seed_list[opt]
        value_list = numpy.array(self.value_lists[opt])
        n_mesh_points = self.optimisation["inverse_n_per_dimension"] # greater than 1
        min_list = numpy.amin(value_list, 0)
        max_list = numpy.amax(value_list, 0)
        step_list = (max_list-min_list)/(n_mesh_points-1)
        n_steps_list = [n_mesh_points]*4
        print("Building inverse grid", min_list, step_list, n_steps_list, max_list)
        grid = ndgrid.NDGrid.initialise_fixed_spacing(n_steps_list, step_list.tolist(), min_list.tolist())
        value_grid_list = []
        for u in grid.coord_vector(0):
            for v in grid.coord_vector(1):
                for w in grid.coord_vector(2):
                    for x in grid.coord_vector(3):
                        value_grid_list.append([u, v, w, x])
        point = [0., 0., 0., 0.] # field
        point_list = []
        limits = self.get_limits()
        for i, value in enumerate(value_grid_list):
            if i%100 == 0:
                print("\r  ... ", i, end="")
            point = minimiser.get_point(self.map_list[opt], seed_list, value, limits)
            point_list.append(copy.deepcopy(point))
        print()
        inverse_map = polynomial_patch.PolynomialPatch.initialise_from_solve_factory(
                                                        grid,
                                                        point_list,
                                                        fit_order,
                                                        smoothing_order)
        print("Inverse map for opt", opt)
        for i, value in enumerate(value_grid_list):
            print(value, point_list[i], inverse_map.function(value))
        return inverse_map

    def plot_map_1d(self, opt, point, variable, constraints, axes):
        """
        opt is index for optimisation
        point is name of the field that we vary (and plot)
        variable is index of the phase space variable that we plot
        constraints is mapping from field name to value of constrained fields
        """
        self.optimisation = self.config.build_bump_surrogate_model["staged_optimisation"][opt]
        grid = self.grid_list[opt]
        values = self.value_lists[opt]
        map_ = self.map_list[opt]
        seed_list = self.seed_list[opt]
        inverse_map = self.inverse_map_list[opt]
        limits = self.get_limits()
        fields = self.get_fields()
        field_index = fields.index(point)
        track_x_list = grid.coord_vector(field_index)
        inv_x_list = []
        minim_y_list = []
        minim_x_list = []
        track_off_list = [grid.coord_vector(i) for i in range(4) if i != field_index]
        track_y_list = []
        min_y_list = []
        max_y_list = []
        fit_x_list = []
        fit_y_list = []
        for i, x0 in enumerate(track_x_list[:-1]):
            x1 = track_x_list[1:][i]
            fit_x_list += [x0+(x1-x0)/10.0*i for i in range(10)]
        fit_x_list.append(x1)
        for x in fit_x_list:
            fit_point = []
            for field in fields:
                if field in constraints:
                    fit_point.append(constraints[field])
                elif field == point:
                    fit_point.append(x)
            print(fit_point)
            fit_value = map_.function(fit_point)
            if inverse_map == None:
                inv_x_list.append(0.)
            else:
                inverse_point = inverse_map.function(fit_value)
                inv_x_list.append(inverse_point[field_index])
            minimiser_point = minimiser.get_multipoint(map_, seed_list, fit_value, limits)
            minimiser_point = [minimiser.get_point(map_, seed_list, fit_value, limits)]
            for test_point in minimiser_point:
                minim_x_list.append(test_point[field_index])
                minim_y_list.append(fit_value[variable])

            if x in track_x_list:
                track_y_list.append(fit_value[variable])
                min_y_list.append(fit_value[variable])
                max_y_list.append(fit_value[variable])
                for a_list_0 in track_off_list[0]:
                    for a_list_1 in track_off_list[0]:
                        for a_list_2 in track_off_list[0]:
                            fit_point = [a_list_0, a_list_1, a_list_2]
                            fit_point.insert(field_index, x)
                            value = map_.function(fit_point)
                            min_y_list[-1] = min(min_y_list[-1], value[variable])
                            max_y_list[-1] = max(max_y_list[-1], value[variable])
            fit_y_list.append(fit_value[variable])

        print("Verse  ", fit_x_list)    
        print("Inverse", inv_x_list)    
        print("Minim  ", minim_x_list)    
        x_axis = point.replace("__", "").replace("_", " ")
        y_axis = ["$x$ [mm]", "$p_{x}$ [MeV/c]", "$y$ [mm]", "$p_{y}$ [MeV/c]"][variable]
        axes.plot(fit_x_list, fit_y_list)
        #axes.plot(inv_x_list, fit_y_list, linestyle="-")
        axes.plot(minim_x_list, minim_y_list, linestyle=" ", marker="x")
        axes.plot(track_x_list, track_y_list, linestyle=" ", marker="o")
        axes.plot(track_x_list, min_y_list, linestyle=" ", marker="o")
        axes.plot(track_x_list, max_y_list, linestyle=" ", marker="o")
        axes.set_xlabel(x_axis+" [T]", fontsize=self.f_size)
        axes.set_ylabel(y_axis, fontsize=self.f_size)

    def plots(self):
        fig_list = []
        range_ = [None]*4
        for i, opt in enumerate(self.config.build_bump_surrogate_model["staged_optimisation"]):
            self.optimisation = opt
            fields = self.get_fields()
            figure = matplotlib.pyplot.figure(figsize=(20, 10))
            for j, field in enumerate(fields):
                axes = figure.add_subplot(2, 2, j+1)
                defaults = self.config.build_bump_surrogate_model["closed_orbit"]
                range1 = defaults[0], defaults[0]+10.
                range2 = defaults[2]-5, defaults[2]+5.
                cm = self.plot_foil_to_var(i, 0, range1, 2, range2, defaults, field, axes)
                figure.colorbar(cm)
            print("Show\n\n")
            matplotlib.pyplot.show(block=False)


    def dont_do(self):
        while False:
                constraints = {}
                for field in fields:
                    if field != plot_field:
                        constraints[field] = 0.0
                figure = matplotlib.pyplot.figure(figsize=(20, 10))
                title = plot_field.replace("__", "")
                print(title)
                figure.suptitle(title)
                fig_dict = {
                    "title":title,
                    "figure":figure,
                    "axes":[]
                }
                for j in range(4):
                    axes = figure.add_subplot(2, 2, j+1)
                    self.plot_map_1d(i, plot_field, j, constraints, axes)
                    fig_dict["axes"].append(axes)
                    if range_[j] == None:
                        range_[j] = list(axes.get_ylim())
                    else:
                        a_range = axes.get_ylim()
                        range_[j][0] = min(range_[j][0], a_range[0])
                        range_[j][1] = max(range_[j][1], a_range[1])
                fig_list.append(fig_dict)
        for fig_dict in fig_list:
            for j, axis in enumerate(fig_dict["axes"]):
                axis.set_ylim(range_[j])
        for fig_dict in fig_list:
            title = fig_dict["title"]
            name = self.config.run_control["output_dir"]+"/"+title+".png"
            print("Saving to", name)
            fig_dict["figure"].savefig(name)

    def plot_foil_to_var(self, opt, foil_var1, foil_range1, foil_var2, foil_range2, foil_defaults, target_var, axes):
        n1_steps = 20
        n2_steps = 20
        fields = self.get_fields()
        field_index = fields.index(target_var)

        limits = self.get_limits()
        print("Using limits", limits)
        seed_list = self.seed_list[opt]
        map_ = self.map_list[opt]
        delta1 = float(foil_range1[1]-foil_range1[0])/n1_steps
        delta2 = float(foil_range2[1]-foil_range2[0])/n2_steps
        var1_list = [i*delta1+foil_range1[0] for i in range(n1_steps+1)]
        var2_list = [i*delta2+foil_range2[0] for i in range(n2_steps+1)]
        z_list = [None]*(n2_steps+1)
        print(var1_list)
        print(var2_list)
        for i, var2 in enumerate(var2_list):
            z_list[i] = [None]*(n1_steps+1)
            print("\r", i, "/", len(var2_list), end="")
            sys.stdout.flush()
            for j, var1 in enumerate(var1_list):
                fit_value = copy.deepcopy(foil_defaults)
                fit_value[foil_var1] = var1
                fit_value[foil_var2] = var2
                try:
                    minimiser_point = minimiser.get_point(map_, seed_list, fit_value, limits)
                    z_list[i][j] = minimiser_point[field_index]
                except ValueError:
                    z_list[i][j] = 0.
        name = target_var.replace("__", "").replace("_", " ")
        cm = axes.contourf(var1_list, var2_list, z_list)
        axes.set_title(name)
        return cm

    dummy_run = False

def main(config):
    bump_model = BuildBumpSurrogateModel(config)
    bump_model.track_map()
    bump_model.build_map()
    bump_model.build_inverse_map()
    bump_model.plots()
    matplotlib.pyplot.show(block=False)
    input()

