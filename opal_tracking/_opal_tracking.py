#This file is a part of xboa
#
#xboa is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#xboa is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with xboa in the doc folder.  If not, see 
#<http://www.gnu.org/licenses/>.

"""
\namespace _opal_tracking
"""

import time
import tempfile
import subprocess
import os
import glob
import math
import sys
import shutil
import h5py

from xboa import common
from xboa.hit import Hit

from xboa.tracking import TrackingBase 

class StoreDataInMemory(object):
    def __init__(self, config):
        self.ignore = config.tracking["ignore_events"]
        self.verbose = config.tracking["verbose"]
        self.dt_tolerance = config.tracking["dt_tolerance"]
        self.station_dt_tolerance = config.tracking["station_dt_tolerance"]
        self.hit_dict_of_lists = {}
        try:
            self.coordinate_transform = \
                  self.coord_dict[config.tracking["analysis_coordinate_system"]]
        except ValueError:
            self.coordinate_transform = "azimuthal"

    def process_hit(self, event, hit):
        if event in self.ignore:
            return
        if not event in self.hit_dict_of_lists:
            self.hit_dict_of_lists[event] = []
        self.hit_dict_of_lists[event].append(hit)

    def clear(self):
        self.hit_dict_of_lists = {}

    def azimuthal_coordinate_transform(self, hit_list_of_lists):
        tmp_hit_list_of_lists = []
        for hit_list in hit_list_of_lists:
            tmp_hit_list = []
            for hit in hit_list:
                x = hit["x"]
                y = hit["z"]
                px = hit["px"]
                py = hit["pz"]
                phi = math.atan2(y, x)
                hit["x"] = + x*math.cos(phi) + y*math.sin(phi)
                hit["z"] = - x*math.sin(phi) + y*math.cos(phi)
                hit["px"] = + px*math.cos(phi) + py*math.sin(phi)
                # go through file line by line reading hit data
                hit["pz"] = + px*math.sin(phi) - py*math.cos(phi)
                hit.mass_shell_condition("energy")
                tmp_hit_list.append(hit)
            tmp_hit_list_of_lists.append(tmp_hit_list)
        return tmp_hit_list_of_lists

    def dt_cut(self, hit_list_of_lists):
        if self.dt_tolerance == None:
            return hit_list_of_lists
        tmp_hit_list_of_lists = []
        cut_count = 0
        for i, hit_list in enumerate(hit_list_of_lists):
            if len(hit_list) == 0:
                tmp_hit_list_of_lists.append([])
                continue
            tmp_hit_list = [hit_list[0]]
            for j, hit_1 in enumerate(hit_list[1:]):
                hit_0 = tmp_hit_list[-1]
                if (hit_1["t"] - hit_0["t"]) < self.dt_tolerance:
                    if self.verbose > 30:
                        print("Cut event", i, "hit", j+1, "with t", hit_1["t"], "compared to", hit_0["t"])
                    cut_count += 1
                    continue
                tmp_hit_list.append(hit_1)
            tmp_hit_list_of_lists.append(tmp_hit_list)
        if self.verbose > 10:
            print("Cut", cut_count, "hits using dt_tolerance", self.dt_tolerance)
        return tmp_hit_list_of_lists

    def reallocate_stations(self, list_of_list_of_hits):
        if self.station_dt_tolerance <= 0.0:
            return list_of_list_of_hits
        max_station = [max([hit['station'] for hit in hit_list]) for hit_list in list_of_list_of_hits]
        max_station = max(max_station)+1
        for hit_list in list_of_list_of_hits:
            if len(hit_list) == 0:
                return
            hit_0 = hit_list[0]
            for hit_1 in hit_list[1:]:
                if hit_1['t'] - hit_0['t'] > self.station_dt_tolerance:
                    hit_1['station'] = hit_0['station']+max_station
                    hit_0 = hit_1
        return list_of_list_of_hits

    def no_coordinate_transform(self, hit_list_of_lists):
        return hit_list_of_lists

    def opal_coordinate_transform(self, hit_list_of_lists):
        for hit_list in hit_list_of_lists:
            for hit in hit_list:
                hit_copy = hit.deepcopy()
                for key_in, key_out in self.opal_dict.items():
                    hit[key_out] = hit_copy[key_in]
                hit.mass_shell_condition("energy")
        return hit_list_of_lists

    def finalise(self):
        # convert from a dict of list of hits to a list of list of hits
        # one list per event
        # each list contains one hit per station
        events = sorted(self.hit_dict_of_lists.keys())
        hit_list_of_lists = [self.hit_dict_of_lists[ev] for ev in events]
        # sort by time within each event
        for i, hit_list in enumerate(hit_list_of_lists):
            hit_list_of_lists[i] = sorted(hit_list, key = lambda hit: hit['t'])
        hit_list_of_lists = self.coordinate_transform(self, hit_list_of_lists)
        hit_list_of_lists = self.dt_cut(hit_list_of_lists)
        hit_list_of_lists = self.reallocate_stations(hit_list_of_lists)
        self.last = hit_list_of_lists
        self.hit_dict_of_lists = {}
        return hit_list_of_lists

    coord_dict = {
        "none":no_coordinate_transform,
        "opal":opal_coordinate_transform,
        "azimuthal":azimuthal_coordinate_transform,
    }

    opal_dict = {"x":"z", "z":"x", "px":"pz", "pz":"px"}

class OpalTracking(TrackingBase):
    """
    Provides an interface to OPAL tracking routines for use by xboa.algorithms
    """
    def __init__(self, lattice_filename, beam_filename, reference_hit, output_filename, opal_path, log_filename = None, n_cores = 1, mpi = None):
        """
        Initialise OpalTracking routines
        - lattice_filename is the Opal lattice file that OpalTracking will use
        - beam_filename is a filename that OpalTracking will overwrite when it
          runs the tracking, putting in beam data
        - reference_hit defines the centroid for the tracking; this should
          correspond to 0 0 0 0 0 0 in the beam file
        - output_filename the name of the PROBE file that OpalTracking will
          read to access output data; wildcards (*, ?) are allowed, in which
          case all files matching the wildcards will be loaded
        - opal_path path to the OPAL executable
        - allow_duplicate_station when evaluates to False, OpalTracking will
          discard duplicate stations on the same event
        - log_filename set to a string file name where OpalTracking will put the 
          terminal output from the opal command; if None, OpalTracking will make
          a temp file
        """
        self.verbose = True
        self.beam_filename = beam_filename
        self.lattice_filename = lattice_filename
        if type(output_filename) == type(""):
            output_filename = {output_filename:(None, None)}
        elif type(output_filename) == type([]):
            output_filename = dict([(name, (None, None)) for name in output_filename])
        elif type(output_filename) == type({}):
            pass
        elif output_filename == None:
            output_filename = None
        else:
            raise(RuntimeError("Did not understand filename type "+str(output_filename)))
        self.output_name_dict = output_filename
        self.opal_path = opal_path
        if not os.path.isfile(self.opal_path):
            raise RuntimeError(str(self.opal_path)+" does not appear to exist."+\
                  " Check that this points to the opal executable.")
        self.ref = reference_hit
        self.last = None
        self.pass_through_analysis = None
        self.station_dt_tolerance = 500.0
        self.do_tracking = True
        self.log_filename = log_filename
        if self.log_filename == None:
            self.log_filename = tempfile.mkstemp()[1]
        self.n_cores = n_cores
        self.mpi = mpi
        self.flags = []
        self.clear_path = None
        self.min_track_number = 1 # minimum number of tracks
        self._read_probes = self._read_ascii_probes
        self.name_dict = {}

    def get_name_dict(self):
        """
        Get a dict of names by globbing self.output_name_dict
        """
        name_dict = {}
        if self.output_name_dict == None:
            return None
        # expand names by globbing each item in output_name_dict, while keeping
        # the station mapping
        for name in self.output_name_dict.keys():
            station, ev = self.output_name_dict[name]
            name_list = glob.glob(name)
            for name in name_list:
                name_dict[name] = (station, ev)
        # if station is None, assign a station
        used_station_list = list(name_dict.values())
        default_station = 0
        for name in sorted(name_dict.keys()):
            station, ev = name_dict[name]
            if (station, ev) == (None, None):
                while default_station in used_station_list:
                    default_station += 1
                name_dict[name] = (default_station, 0)
                used_station_list.append(default_station)
        self.name_dict = name_dict
        return name_dict

    def get_name_list(self):
        """
        Get a list of names by globbing self.output_name_dict
        """
        return sorted(self.get_name_dict().keys())

    def save(self, target_dir, save_all = True, clear_dir = False):
        """
        Save to 
        """
        if os.path.exists(target_dir):
            if clear_dir: 
                shutil.rmtree(target_dir)
                os.makedirs(target_dir)
        else:
            os.makedirs(target_dir)
        if save_all:
            dir_list = []
            for tgt in self.get_name_list():
                dir_list.append(os.path.split(tgt)[0])
            dir_list = set(dir_list)
            name_list = []
            for a_dir in dir_list:
                name_list += glob.glob(os.path.join(a_dir, "*"))
        else:
            name_list = self.get_name_list()
        if self.verbose > -10:
            print("Saving", len(name_list), "files to", target_dir)
        for src in name_list:
            target = os.path.join(target_dir, src)
            if os.path.isdir(src):
                if os.path.isdir(target):
                    shutil.rmtree(target)
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, target)

    def cleanup(self):
        """
        Delete output files (prior to tracking)
        """
        if self.clear_path == None:
            clear_files = self.get_name_list()
        else:
            clear_files = glob.glob(self.clear_path)
        for a_file in clear_files:
            os.unlink(a_file)

    def track_one(self, hit):
        """
        Track one hit through Opal

        Returns a list of hits, sorted by time.
        """
        return self.track_many([hit])[0]
        
    def track_many(self, list_of_hits):
        """
        Track many hits through Opal

        Returns a list of lists of hits; each list of hits corresponds to a
        track, defined by probe "id" field. Output hits are sorted by time 
        within each event.
        """
        if self.do_tracking:
            self._tracking(list_of_hits)
        if self.verbose > 30:
            print("Read probes")
        hit_list_of_lists = self._read_probes()
        if self.verbose > 30:
            lengths = [len(hit_list) for hit_list in hit_list_of_lists]
            print("Read", len(hit_list_of_lists), "tracks with length", lengths, ". Saving.")
        if self.verbose > 30:
            print("Return")
        return hit_list_of_lists

    def open_subprocess(self):
        command = [self.opal_path, self.lattice_filename]+self.flags
        will_do_bsub = False
        if self.mpi != None:
            try:
                subprocess.check_call(["bsub", "-V"])
                will_do_bsub = True
                bsub_command = ["bsub",
                                "-n", str(self.n_cores),
                                "-q", 'scarf-ibis',
                                "-W", "24:00",
                                "-o", self.log_filename,
                                "-K", ]
                command = bsub_command+[self.mpi]+command
            except OSError:
                command = [self.mpi, "-n", str(self.n_cores)]+command
        if will_do_bsub:
            log = open("scarf.log", "w")
        else:
            log = open(self.log_filename, "w")

        proc = subprocess.Popen(command,
                                stdout=log,
                                stderr=subprocess.STDOUT)
        return proc

    def _tracking(self, list_of_hits):
        if self.verbose:
            print("Tracking in dir", os.getcwd(),
                  "\n   using logfile", self.log_filename)
        open(self.lattice_filename).close() # check that lattice exists
        metres, GeV, seconds = common.units["m"], common.units["GeV"], common.units["s"]
        p_mass = common.pdg_pid_to_mass[2212]
        fout = open(self.beam_filename, "w")
        # OPAL goes into odd modes if there are < 2 entries in the beam file
        while len(list_of_hits) > 0 and len(list_of_hits) < self.min_track_number:
            list_of_hits.append(list_of_hits[-1])
        print(len(list_of_hits), file=fout)
        for i, hit in enumerate(list_of_hits):
            if self.verbose:
                if i == 0:
                    print('           ', end=' ')
                    for key in self.print_keys:
                        print(key.ljust(8), end=' ')
                    print('\n    ref ...', end=' ')
                    for key in self.print_keys:
                        print(str(round(self.ref[key], 3)).ljust(8), end=' ')
                if i < 1 or i == len(list_of_hits)-1:
                    print('\n    hit ...', end=' ')
                    for key in self.print_keys:
                        print(str(round(hit[key], 3)).ljust(8), end=' ')
                    print()
                if i == 1 and len(list_of_hits) > 2:
                    print("<", len(list_of_hits)-2, " more hits >")
            x = (hit["x"]-self.ref["x"])/metres
            y = (hit["y"]-self.ref["y"])/metres
            t = hit["t"]/seconds
            px = (hit["px"]-self.ref["px"])/p_mass
            py = (hit["py"]-self.ref["py"])/p_mass
            pz = (hit["pz"]-self.ref["pz"])/p_mass
            if abs(hit["z"]) > 1e-9:
                print("Attempt to make hit with z non-zero", hit["z"])
                raise RuntimeError("z is not available in distribution input")
            print(x, px, y, py, t, pz, file=fout)
        fout.close()
        self.cleanup()
        old_time = time.time()
        proc = self.open_subprocess()
        proc.wait()
        if self.verbose:
            print("Ran for", time.time() - old_time, "s")
        # returncode 1 -> particle fell out of the accelerator
        if proc.returncode != 0 and proc.returncode != 1:
            try:
                raise RuntimeError("OPAL quit with non-zero error code "+\
                                   str(proc.returncode)+". Review the log file: "+\
                                   os.path.join(os.getcwd(), self.log_filename))
            except:
                sys.excepthook(*sys.exc_info())

    def set_file_format(self, file_format):
        if file_format == "ascii":
            self._read_probes = self._read_ascii_probes
        elif file_format == "hdf5":
            self._read_probes = self._read_h5_probes
        else:
            raise ValueError("Did not recognise Probe file format '"+str(file_format)+\
                             "'. Options are 'ascii' or 'hdf5'")

    def _read_ascii_probes(self):
        # loop over files in the glob, read events and sort by event number
        file_dict = self.get_name_dict()
        if file_dict == None:
            return []
        if len(file_dict) == 0:
            name_list = str(self.output_name_list)
            raise IOError("Failed to load any probes from "+name_list)
        fin_list = []
        for file_name, station in sorted(file_dict.items()):
            try:
                fin_list.append((open(file_name), station))
            except OSError:
                pass
        line = "0"
        line_number = 0
        while line != "" and len(fin_dict) > 0:
            line_number += 1
            for fin, station in fin_list:
                line = fin.readline()
                if line == "":
                    break
                try:
                    event, hit = self.read_one_ascii_line(line, station)
                    self.pass_through_analysis.process_hit(event, hit)
                except ValueError:
                    pass
                    # OPAL accumulates LOSS over many runs, but this may not be
                    # desired; so clear
                    #pass_through_analysis.clear()
        self.last = self.pass_through_analysis.finalise()
        return self.last

    def read_one_ascii_line(self, line, station): 
        words = line.split()
        hit_dict = {}
        for key in "pid", "mass", "charge":
            hit_dict[key] = self.ref[key]
        for i, key in enumerate(["x", "z", "y"]):
            hit_dict[key] = float(words[i+1])*1000.
        for i, key in enumerate(["px", "pz", "py"]):
            hit_dict[key] = float(words[i+4])*self.ref["mass"]
        hit_dict["event_number"] = int(words[7])
        hit_dict["station"] = station
        hit_dict["t"] = float(words[9])
        hit = Hit.new_from_dict(hit_dict, "energy")
        return hit_dict["event_number"], hit

    def _read_h5_probes(self):
        # loop over files in the glob, read events and sort by event number
        file_dict = self.get_name_dict()
        if self.verbose:
            print("Found following files", str(file_dict))
        if file_dict == None:
            return []
        if len(file_dict) == 0:
            name_list = str(self.output_name_dict)
            raise IOError("Failed to load any probes from "+name_list)
        file_list = []
        for file_name, station in sorted(file_dict.items()):
            try:
                file_list.append((h5py.File(file_name, 'r'), station, file_name))
                print("Opened", file_name)
            except OSError:
                pass
        hit = ""
        for fin, tup, file_name in file_list:
            station, ev = tup
            try:
                hit_generator = self.generate_h5_step(fin, ev, station)
            except KeyError:
                sys.excepthook(*sys.exc_info())
                print("Failed in", file_name)
            for event, hit in hit_generator:
                self.pass_through_analysis.process_hit(event, hit)
        self.last = self.pass_through_analysis.finalise()
        return self.last

    def generate_h5_step(self, h5_file, ev, station):
        for key in h5_file.keys():
            if key[:5] != "Step#":
                if self.verbose > 10:
                    print("Skipping", key)
                continue
            n_steps = len(h5_file[key]["x"])
            if self.verbose > 10:
                print("Found", key, "in", n_steps, "events in", h5_file.filename)
                print(" fmt", "  ev ", end="")
                for var in self.print_keys:
                    print(var.ljust(8), end=' ')
                print()
            h5_step = h5_file[key]
            for i in range(n_steps):
                hit_dict = {}
                for key in "pid", "mass", "charge":
                    hit_dict[key] = self.ref[key]
                for h5_key, xboa_key in self.h5_key_to_xboa_key.items():
                    hit_dict[xboa_key] = h5_step[h5_key][i]
                    if xboa_key in self.units:
                        hit_dict[xboa_key] *= self.units[xboa_key]
                for key in ["px", "py", "pz"]:
                    hit_dict[key] *= self.ref["mass"]
                hit_dict["station"] = station
                hit = Hit.new_from_dict(hit_dict, "energy")
                if self.verbose > 10:
                    print("  h5", format(hit["event_number"], "4d"), end=' ')
                    for key in self.print_keys:
                        print(str(round(hit[key], 3)).ljust(8), end=' ')
                    print()
                hit["event_number"] += ev
                yield hit["event_number"], hit

    h5_key_to_xboa_key = {"y":"x", "z":"y", "x":"z", "time":"t",
                          "py":"px", "pz":"py", "px":"pz", "id":"event_number"}
    units = {"x":1000., "y":1000., "z":1000.}

    print_keys = ['x', 'y', 'z', 'px', 'py', 'pz', 'kinetic_energy', 't']
