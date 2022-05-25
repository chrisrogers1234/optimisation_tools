
import glob
import sys
import os
import json
import time
import copy

import numpy

import xboa.common as common
from xboa.hit import Hit
from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.utils import utilities
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix


class DAFinder(object):
    def __init__(self, config):
        self.closed_orbit_file_name = os.path.join(config.run_control["output_dir"], config.find_closed_orbits["output_file"])
        self.da_file_name = os.path.join(config.run_control["output_dir"], config.find_da["get_output_file"])
        self.scan_file_name = os.path.join(config.run_control["output_dir"], config.find_da["scan_output_file"]) 
        self.config = config
        self.run_dir =  os.path.join(config.run_control["output_dir"], config.find_da["run_dir"])
        self.co_list = self.load_closed_orbits()
        self.ref_hit = None
        self.min_delta = config.find_da["min_delta"]
        self.max_delta = config.find_da["max_delta"]
        self.data = []
        self.fout_scan_tmp = None
        self.fout_get_tmp = None
        self.required_n_hits = config.find_da["required_n_hits"]
        self.max_iterations = config.find_da["max_iterations"]
        self.save_dir = os.path.join(self.config.run_control["output_dir"],
                                     self.config.find_da["save_dir"])
        print("Saving tracking output to", self.save_dir)
        self.tracking = self.setup()
        self.var_list = ["x", "px", "y", "py"]
        DecoupledTransferMatrix.det_tolerance = 1.


    def get_all_da(self, co_index_list, seed_x, seed_y):
        for test_seed in self.co_list:
            if co_index_list == None:
                co_index_list = list(range(len(test_seed)))
            for i in range(len(test_seed)):
                try:
                    co_element = test_seed[i]
                    print("Finding da for element", i)
                except KeyError:
                    print("Failed to find index", i, "in co_list of length", len(co_list))
                    continue
                if seed_x != None and seed_x > 0.:
                    co_element['x_da'] = self.get_da(co_element, 'x', seed_x)
                if seed_y != None and seed_y > 0.:
                    co_element['y_da'] = self.get_da(co_element, 'y', seed_y)
                print(json.dumps(co_element), file=self.fout_get())
                self.fout_get().flush()

    def da_all_scan(self, co_index_list, x_list, y_list):
        if co_index_list == None:
            co_index_list = list(range(len(self.co_list)))
        for i in co_index_list:
            try:
                co_element = self.co_list[i]
                print(co_element)
                print("Scanning da for element", i)
            except KeyError:
                print("Failed to find index", i, "in co_list of length", len(co_list))
                continue
            co_element['da_scan'] = self.da_scan(co_element, x_list, y_list)

    def load_closed_orbits(self):
        fin = open(self.closed_orbit_file_name)
        co_list = [json.loads(line) for line in fin.readlines()]
        print("Loaded", len(co_list), "closed orbits")
        return co_list

    def setup(self):
        self.tmp_dir = "./"
        try:
            os.makedirs(self.run_dir)
        except OSError: # maybe the dir already exists
            pass
        os.chdir(self.run_dir)
        print("Running in", os.getcwd())
        self.opal_exe = os.path.expandvars("${OPAL_EXE_PATH}/opal")

    def reference(self, hit_dict):
        """
        Generate a reference particle
        """
        hit = Hit.new_from_dict(hit_dict)
        hit["x"] = 0.
        hit["px"] = 0.
        return hit

    def setup_tracking(self, co_element):
        overrides = self.config.find_da["subs_overrides"]
        overrides["__n_particles__"] = 3
        utilities.do_lattice(self.config,
                             co_element["substitutions"],
                             overrides)
        self.tracking = utilities.setup_tracking(
                                self.config,
                                self.config.find_da["probe_files"],
                                co_element["substitutions"]["__energy__"])

    def new_seed(self):
        if self.data[-1][0] < self.min_delta: # reference run?
            return None
        if self.test_pass(*self.data[-1]): # upper limit is okay; keep going up
            if self.data[-1][0] > self.max_delta: # too big; give up
                return None
            return self.data[-1][0]*2.
        elif not self.test_pass(*self.data[0]): # lower limit is bad; try going down
            if abs(self.data[0][0]) < self.min_delta:
                return None
            return self.data[0][0]/2.
        else:
            for i, item in enumerate(self.data[1:]):
                if not self.test_pass(*item):
                    break
            if abs(self.data[i][0]-self.data[i+1][0]) < self.min_delta:
                return None
            return (self.data[i][0]+self.data[i+1][0])/2.

    def test_pass(self, seed, hits_list):
        return len(hits_list) >= self.required_n_hits

    def time_cut(self, hits_list):
        dt = 0
        ref_dt = 100.
        if len(hits_list) > 2:
            ref_dt = hits_list[2]['t'] - hits_list[1]['t']
        tmp_hits_list = [hits_list[0]]
        dt_tolerance = self.config.find_da["dt_tolerance"]
        for i, hit_1 in enumerate(hits_list[1:]):
            hit_0 = hits_list[i]
            dt = (hit_1['t'] - hit_0['t'])
            ratio = abs(dt/ref_dt-1)
            # print(i, "dt:", dt, "ref:", ref_dt, "ratio:", ratio, "tol:", dt_tolerance)
            if ratio < dt_tolerance:
                tmp_hits_list.append(hit_1)
        return tmp_hits_list

    def events_generator(self, co_element, x_list, y_list):
        co_hit = co_element["hits"][0]
        for x in x_list:
            for y in y_list:
                a_hit = co_hit.deepcopy()
                a_hit['x'] = co_hit['x'] + x
                a_hit['px'] = co_hit['px']
                a_hit['y'] += y
                yield {"x":x, "y":y}, a_hit

    def da_scan(self, co_element, x_list, y_list):
        self.setup_tracking(co_element)
        gen = self.events_generator(co_element, x_list, y_list)
        self.data = []
        finished = False
        while not finished:
            event_list = []
            track_list = []
            try:
                while len(event_list) < 1:
                    track, event = next(gen)
                    track_list.append(track)
                    event_list.append(event)
            except StopIteration:
                finished = True
            if len(event_list) == 0:
                break
            many_tracks = self.tracking.track_many(event_list)
            for i, hits in enumerate(many_tracks):
                t_hits = self.time_cut(hits)
                print("Tracked", len(t_hits), "/", len(hits), 
                      "hits passing time cut with track", track_list[i],
                      "first event x px y py", hits[0]["x"], hits[0]["px"], hits[0]["y"], hits[0]["py"])
                self.data.append([track_list[i], [a_hit.dict_from_hit() for a_hit in t_hits]])
        print(json.dumps(self.data), file=self.fout_scan())
        self.fout_scan().flush()

    def seed_to_hit(self, co_element, seed):
        closed_orbit = Hit.new_from_dict(co_element["ref_track"][0])
        if self.config.find_da["decoupled"]:
            tm = copy.deepcopy(co_element["tm"])
            for i, row in enumerate(tm):
                tm[i] = row[1:5]
            tm = DecoupledTransferMatrix(tm)
            seed = tm.coupled(seed)
        a_hit = closed_orbit.deepcopy()
        for i, x in enumerate(seed):
            a_hit[self.var_list[i]] += x
        return a_hit

    def hit_to_seed(self, co_element, hit):
        seed = []
        closed_orbit = Hit.new_from_dict(co_element["ref_track"][0])
        for var in self.var_list:
            seed.append(hit[var]-closed_orbit[var])
        print("Tracked hit; coupled", seed, end=" ")
        if self.config.find_da["decoupled"]:
            tm = copy.deepcopy(co_element["tm"])
            for i, row in enumerate(tm):
                tm[i] = row[1:5]
            tm = DecoupledTransferMatrix(tm, True)
            seed = tm.decoupled(seed)
        print("decoupled", seed)
        return seed

    def test_propagate_seed(self, co_element, seed):
        tm = copy.deepcopy(co_element["tm"])
        for i, row in enumerate(tm):
            tm[i] = row[1:5]
        tm = DecoupledTransferMatrix(tm)
        cp_seed = tm.coupled(seed)
        print("Propagating:", cp_seed, "from decoupled", seed)
        for i in range(10):
            cp_seed = numpy.dot(tm.m, cp_seed)
            print("Propagated: ", cp_seed, "as decoupled  ", tm.decoupled(cp_seed))

    def get_da(self, co_element, axis, seed_x):
        is_ref = abs(seed_x) < 1e-6
        self.setup_tracking(co_element)
        self.data = []
        axis_i = self.var_list.index(axis)
        co_delta = [0., 0., 0., 0.]
        iteration = 0
        while seed_x != None and iteration < self.max_iterations:
            co_delta[axis_i] = seed_x
            my_time = time.time()
            a_hit = self.seed_to_hit(co_element, co_delta)
            try:
                hits = self.tracking.track_many([a_hit])[-1]
                print("Source seed:", co_delta)
                for hit in hits:
                    test_seed = self.hit_to_seed(co_element, hit)
                self.test_propagate_seed(co_element, co_delta)

            except RuntimeError:
                sys.excepthook(*sys.exc_info())
                print("Never mind, keep on going...")
            if self.config.find_da["save_dir"]:
                save_dir = os.path.join(self.save_dir, axis+"_"+str(iteration))               
                self.tracking.save(save_dir)
            t_hits = self.time_cut(hits)
            self.data.append([co_delta[axis_i], [a_hit.dict_from_hit() for a_hit in t_hits]])
            self.data = sorted(self.data)
            print("Axis", axis, "Seed", seed_x,
                  "Number of cells passing time cut/total hits",
                  len(t_hits), "/", len(hits), 
                  "in", time.time() - my_time, "[s]")
            sys.stdout.flush()
            seed_x = self.new_seed()
            if is_ref:
                seed_x = None
            iteration += 1
        return self.data

    def fout_scan(self):
        if self.fout_scan_tmp == None:
            file_name = self.scan_file_name+".tmp"
            self.fout_scan_tmp = open(file_name, "w")
            print("Opened file", file_name)
        return self.fout_scan_tmp

    def fout_get(self):
        if self.fout_get_tmp == None:
            file_name = self.da_file_name+".tmp"
            self.fout_get_tmp = open(file_name, "w")
            print("Opened file", file_name)
        return self.fout_get_tmp

def main(config):
    finder = DAFinder(config)
    #finder.da_all_scan(config.find_da["row_list"], config.find_da["scan_x_list"], config.find_da["scan_y_list"])
    finder.get_all_da(config.find_da["row_list"], config.find_da["x_seed"], config.find_da["y_seed"], )
    print("Done find da")
    sys.stdout.flush()
    return
