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

import copy
import time
import tempfile
import subprocess
import os
import glob
import math
import sys
import shutil


import ROOT
from xboa import common
import xboa.hit
from xboa.hit import Hit
from xboa.bunch import Bunch
from xboa.tracking import TrackingBase

IMPORTED_BDSIM_DEFINITIONS = False
def import_bdsim_definitions():
    global IMPORTED_BDSIM_DEFINITIONS
    if IMPORTED_BDSIM_DEFINITIONS:
        print("Already imported bdsim definitions")
        return
    import ROOT
    if "BDSIM_LIB_PATH" not in os.environ:
        raise ImportError("BDSIM_LIB_PATH must be an environment variable")
    bdsim_path = os.environ["BDSIM_LIB_PATH"]
    if not os.path.isdir(bdsim_path):
        raise ImportError("BDSIM_LIB_PATH must point at a directory")
    print("Loading from", bdsim_path)
    ROOT.gSystem.AddDynamicPath(bdsim_path)
    print("    Loading libbdsimRootEvent")
    bdsLoad = ROOT.gSystem.Load("libbdsimRootEvent")
    print("    Loading librebdsim")
    reLoad  = ROOT.gSystem.Load("librebdsim")
    print("Done loading ROOT libs")
    IMPORTED_BDSIM_DEFINITIONS = True

class BDSIMTracking(TrackingBase):
    """
    Provides an interface to BDSIM tracking routines
    """
    def __init__(self, lattice_filename, beam_filename, reference_hit, output_filename, bdsim_path, log_filename = None):
        """
        Initialise BDSIMTracking routines
        - lattice_filename is the BDSIM lattice file that BSIMTracking will use
        - beam_filename is a filename that BDSIMTracking will overwrite when it
          runs the tracking, putting in beam data
        - reference_hit defines the centroid for the tracking; this should
          correspond to 0 0 0 0 0 0 in the beam file
        - output_filename the name of the output file that BDSIMTracking will
          read to access output data; glob wildcards (*, ?) are allowed, in
          which case all files matching the wildcards will be loaded; if a list
          is used, BDSIM will glob each element in the list
        - bdsim_path path to the BDSIM executable
        - allow_duplicate_station when evaluates to False, BDSIMTracking will
          discard duplicate stations on the same event
        - log_filename set to a string file name where BDSIMTracking will put the
          terminal output from the opal command; if None, BDSIMTracking will make
          a temp file
        """
        import_bdsim_definitions()
        self.verbose = True
        self.beam_filename = beam_filename
        self.lattice_filename = lattice_filename
        if type(self.lattice_filename) == type([]):
            self.lattice_filename = self.lattice_filename[0]
        self.output_filename_list = output_filename
        self.output_format = "bdsim_root"
        if type(self.output_filename_list) == type(""):
            self.output_filename_list = [self.output_filename_list]
        self.bdsim_path = bdsim_path
        if not os.path.isfile(self.bdsim_path):
            raise RuntimeError(str(self.bdsim_path)+" does not appear to exist."+\
                  " Check that this points to the opal executable.")
        self.ref = reference_hit
        self.ignore_reference = True
        self.last = None
        self.pass_through_analysis = None
        self.do_tracking = True
        self.log_filename = log_filename
        self.station_to_z_dict = None
        if self.log_filename == None:
            self.log_filename = tempfile.mkstemp()[1]
        self.flags = ["--batch"]
        self.clear_path = None
        self.min_track_number = 1 # minimum number of tracks
        self.name_dict = {}

    def save(self, target_dir, save_all = True, clear_dir = False):
        """
        Save to a destination for later post-processing
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

    def get_name_list(self):
        file_name_list = []
        for file_name in self.output_filename_list:
            file_name_list += glob.glob(file_name)
        return file_name_list

    def get_name_dict(self):
        return dict([(file_name, None) for file_name in self.get_name_list()])


    def track_one(self, hit):
        """
        Track one hit through Opal

        Returns a list of hits, sorted by time.
        """
        return self.track_many([hit])[0]
        
    def track_many(self, list_of_hits, renumber=1):
        """
        Track many hits through BDSIM
        - list_of_hits: list of type xboa.hit.Hit, the input particles to be tracked
        - renumber: set to an integer to reassign event ids, starting with renumber.
          Set to None to disable renumbering.

        Returns a list of lists of hits; each list of hits corresponds to a
        track, defined by the "evid" field.
        """
        if renumber is not None:
            list_of_hits = [copy.deepcopy(hit) for hit in list_of_hits] # force out shallow copies
            for hit in list_of_hits:
                hit["event_number"] = renumber
                renumber = renumber+1
        if self.do_tracking:
            self._tracking(list_of_hits)
        if self.verbose > 30:
            print("Read probes")
        hit_list_of_lists = self._read_files()
        if self.verbose > 30:
            lengths = [len(hit_list) for hit_list in hit_list_of_lists]
            print("Read", len(hit_list_of_lists), "tracks with length", lengths, ". Saving.")
        if self.verbose > 30:
            print("Return")
        return hit_list_of_lists

    def open_subprocess(self, nevents):
        #self.cleanup()
        command = [self.bdsim_path, f"--file={self.lattice_filename}", f"--ngenerate={nevents}", "--batch"]+self.flags
        log = open(self.log_filename, "w")
        proc = subprocess.Popen(command,
                                stdout=log,
                                stderr=subprocess.STDOUT)
        return proc

    def _tracking(self, list_of_hits):
        with open(self.beam_filename, "w") as fout:
            fout.write('"x [mm]" "px [rad]" "y [mm]" "py [rad]" "z [mm]" "Ek [MeV]" "t [ns]"\n')
            for hit in list_of_hits:
                hit.write_user_formatted(["x","x'","y","y'", "z", "kinetic_energy", "t"], {"x":"mm","x'":"","y":"mm","y'":"", "z":"mm", "kinetic_energy":"MeV", "t":"ns"}, fout)
        old_time = time.time()
        proc = self.open_subprocess(len(list_of_hits))
        proc.wait()
        if self.verbose:
            print("Ran for", time.time() - old_time, "s")
        if proc.returncode != 0:
            try:
                raise RuntimeError("BDSIM quit with non-zero error code "+\
                                   str(proc.returncode)+". Review the log file: "+\
                                   os.path.join(os.getcwd(), self.log_filename))
            except:
                sys.excepthook(*sys.exc_info())

    @classmethod
    def sampler_search(cls, event):
        sampler_list = []
        index = -1
        while True:
            index += 1
            try:
                sampler_list.append(getattr(event, f"s{index}"))
            except AttributeError:
                if index < 10:
                    continue
                elif index > 10:
                    if len(sampler_list) == 0:
                        raise RuntimeError(f"Could not find any samplers in {file_name}")
                    elif index < len(sampler_list)*10:
                        continue
                    else:
                        break
        return sampler_list

    @classmethod
    def bdsim_rezedify(cls, hit, station_to_z_dict):
        if station_to_z_dict == None:
            return
        hit["z"] = station_to_z_dict[hit["station"]]

    @classmethod
    def read_files(cls, file_name_list_of_globs, pass_through_analysis, station_to_z_dict=None, verbose=0):
        mm = xboa.common.units["mm"]
        gev = xboa.common.units["GeV"]
        ns = xboa.common.units["ns"]
        file_name_list = []
        for file_name in file_name_list_of_globs:
            file_name_list += glob.glob(file_name)
        if verbose > 10:
            print(f"Found {len(file_name_list)} files from glob {file_name}")
        for file_name in file_name_list:
            if verbose > 10:
                print("Loading", file_name)
            fin = ROOT.TFile(file_name)
            for event_number, event in enumerate(fin.Get("Event")):
                sampler_list = cls.sampler_search(event)
                if verbose > 100:
                    print(f"Found {len(sampler_list)} samplers in event {event_number}")
                for station, sampler in enumerate(sampler_list):
                    if len(sampler.x) == 0:
                        continue
                    hit_dict = {
                        "x":sampler.x[0]*mm,
                        "y":sampler.y[0]*mm,
                        "z":sampler.z*mm,
                        "t":sampler.T[0]*ns,
                        "px":sampler.xp[0]*sampler.p[0]*gev,
                        "py":sampler.yp[0]*sampler.p[0]*gev,
                        "pz":sampler.zp[0]*sampler.p[0]*gev,
                        "pid":sampler.partID[0],
                        "station":station,
                        "particle_number":sampler.trackID[0],
                        "event_number":event_number
                    }
                    hit_dict["mass"] = xboa.common.pdg_pid_to_mass[abs(hit_dict["pid"])]
                    hit = xboa.hit.Hit.new_from_dict(hit_dict, "energy")
                    cls.bdsim_rezedify(hit, station_to_z_dict)
                    print("station:", station, "ev:", event_number, "z:", hit["z"], "t:",hit["t"] , "E:", hit["energy"] )
                    if verbose > 50:
                        print("Loaded hit", hit_dict)
                        print("Sampler energy was", sampler.energy[0]*gev, hit["energy"])
                    assert(abs(hit["energy"] - sampler.energy[0]*gev) < 1e-4)
                    pass_through_analysis.process_hit(event_number, hit)


    def _read_files(self):
        # loop over files in the glob, read events and sort by event number
        self.read_files(self.output_filename_list, self.pass_through_analysis, self.station_to_z_dict, self.verbose)
        self.last = self.pass_through_analysis.finalise()
        return self.last

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
