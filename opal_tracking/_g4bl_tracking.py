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

from xboa import common
import xboa.hit
from xboa.hit import Hit
from xboa.bunch import Bunch

from xboa.tracking import TrackingBase 

class G4BLTracking(TrackingBase):
    """
    Provides an interface to G4BL tracking routines
    """
    def __init__(self, lattice_filename, beam_filename, reference_hit, output_filename, g4bl_path, log_filename = None):
        """
        Initialise G4BLTracking routines
        - lattice_filename is the G4BL lattice file that G4BLTracking will use
        - beam_filename is a filename that G4BLTracking will overwrite when it
          runs the tracking, putting in beam data
        - reference_hit defines the centroid for the tracking; this should
          correspond to 0 0 0 0 0 0 in the beam file
        - output_filename the name of the output file that G4BLTracking will
          read to access output data; glob wildcards (*, ?) are allowed, in
          which case all files matching the wildcards will be loaded; if a list
          is used, G4BL will glob each element in the list
        - g4bl_path path to the G4BL executable
        - allow_duplicate_station when evaluates to False, G4BLTracking will
          discard duplicate stations on the same event
        - log_filename set to a string file name where G4BLTracking will put the 
          terminal output from the opal command; if None, G4BLTracking will make
          a temp file
        """
        self.verbose = True
        self.beam_filename = beam_filename
        self.beam_format = "g4beamline_bl_track_file"
        self.lattice_filename = lattice_filename
        if type(self.lattice_filename) == type([]):
            self.lattice_filename = self.lattice_filename[0]
        self.output_filename_list = output_filename
        self.output_format = "icool_for009"
        if type(self.output_filename_list) == type(""):
            self.output_filename_list = [self.output_filename_list]
        self.g4bl_path = g4bl_path
        if not os.path.isfile(self.g4bl_path):
            raise RuntimeError(str(self.g4bl_path)+" does not appear to exist."+\
                  " Check that this points to the opal executable.")
        self.ref = reference_hit
        self.ignore_reference = True
        self.last = None
        self.pass_through_analysis = None
        self.do_tracking = True
        self.log_filename = log_filename
        if self.log_filename == None:
            self.log_filename = tempfile.mkstemp()[1]
        self.flags = []
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
        Track many hits through G4BL
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

    def open_subprocess(self):
        command = [self.g4bl_path, self.lattice_filename]+self.flags
        log = open(self.log_filename, "w")
        proc = subprocess.Popen(command,
                                stdout=log,
                                stderr=subprocess.STDOUT)
        return proc

    def _tracking(self, list_of_hits):
        Hit.write_list_builtin_formatted(list_of_hits, self.beam_format, self.beam_filename)
        old_time = time.time()
        proc = self.open_subprocess()
        proc.wait()
        if self.verbose:
            print("Ran for", time.time() - old_time, "s")
        if proc.returncode != 0:
            try:
                raise RuntimeError("G4BL quit with non-zero error code "+\
                                   str(proc.returncode)+". Review the log file: "+\
                                   os.path.join(os.getcwd(), self.log_filename))
            except:
                sys.excepthook(*sys.exc_info())

    def _read_files(self):
        # loop over files in the glob, read events and sort by event number
        file_name_list = []
        for file_name in self.output_filename_list:
            file_name_list += glob.glob(file_name)
        for file_name in file_name_list:
            if self.verbose > 10:
                print("Loading", file_name)
            n_good_lines, n_lines = 0, 0
            fin = open(file_name)
            while fin:
                try:
                    n_lines += 1
                    hit = Hit.new_from_read_builtin(self.output_format, fin)
                    event = hit["event_number"]
                    if self.ignore_reference and event == 0:
                        continue
                    self.pass_through_analysis.process_hit(event, hit)
                    n_good_lines += 1
                except (EOFError, StopIteration):
                    break                    
                except (xboa.hit.BadEventError, ValueError):
                    pass
                except:
                    sys.excepthook(*sys.exc_info())
                    break
            if self.verbose > 10:
                print("Successfull parsed", n_good_lines, "/", n_lines, "lines")
        self.last = self.pass_through_analysis.finalise()
        return self.last
