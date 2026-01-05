"""
Script to find the RF set up; drives find closed orbit
"""

import os
import sys
import copy
import json
import math
import glob

import numpy
import ROOT

import xboa.common
from xboa.hit import Hit
sys.path.insert(1, "scripts")
from optimisation_tools.opal_tracking import OpalTracking
from optimisation_tools.utils import utilities

class TrackBump(object):
    def __init__(self, config):
        self.config = config
        self.tmp_dir = os.path.join(self.config.run_control["output_dir"],
                               self.config.track_bump["run_dir"])
        self.subs = {}
        self.tracking = None
        self.bump_data = []
        self.output = []
        self.injection_orbit = self.config.track_bump["injection_orbit"]
        self.file_index = 0
        self.data_dict = {}


    def clear_dir(self):
        lattice_name = self.config.tracking["lattice_file_out"].split(".")[0]
        for item in glob.glob(self.tmp_dir+"/"+lattice_name+"-trackOrbit*"):
            try:
                print("Clearing", item)
                os.unlink(item)
            except OSError: # maybe the file never existed
                pass
        try:
            os.makedirs(self.tmp_dir)
        except OSError: # maybe the dir already exists
            pass

    def load_files(self):
        fglob = os.path.join(self.config.run_control["output_dir"],
                             self.config.track_bump["input_file"])
        bump_data = []
        print("Loading bump files")
        for fname in sorted(glob.glob(fglob)):
            print("   ", fname)
            with open(fname) as fin:
                str_in = fin.readlines()[-1]
            bump_data.append(json.loads(str_in))
        return bump_data

    def load_bump_parameters(self):
        os.chdir(self.tmp_dir)
        bump_data = self.load_files()
        self.data_dict = {}
        for item in bump_data:
            if "optimisation_stage" in item:
                id_ = item["optimisation_stage"]
            else:
                id_ = {"substitution_index":0, "bump_index":0, "optimisation_stage":0}
            data_key = (id_["substitution_index"],
                        id_["bump_index"])
            if data_key not in self.data_dict:
                self.data_dict[data_key] = {}
            if id_["optimisation_stage"] == self.config.track_bump["foil_optimisation_stage"]:
                self.data_dict[data_key]["proton_orbit"] = \
                                                    self.get_proton_orbit(item)
            if id_["optimisation_stage"] == self.config.track_bump["field_optimisation_stage"]:
                self.data_dict[data_key]["field_list"] = \
                                                    self.get_field_list(item)
            print("Found id", id_)

    def track_bumps(self):
        self.clear_dir()
        self.load_bump_parameters()
        for i, key in enumerate(sorted(self.data_dict.keys())):
            bump = self.data_dict[key]
            if "field_list" not in bump or "proton_orbit" not in bump:
                print("Skipping key", key)
                continue
            self.subs = self.config.substitution_list[key[0]]
            self.print_bump(key, bump)
            track = self.fore_track(bump["field_list"], bump["proton_orbit"])
            if i == self.config.track_bump["injection_bump"]:
                back_subs = self.subs
                self.get_injection_orbit(track)
        print("Back tracking using injection orbit:", self.injection_orbit)
        self.subs = back_subs
        self.back_track(bump["field_list"])


    def get_injection_orbit(self, track):
        if self.injection_orbit != None:
            print("Using user-supplied injection orbit", self.injection_orbit)
            return
        var_list = ["x", "px", "y", "py"] 
        self.injection_orbit = [track[0][0][var] for var in var_list]
        print("Found injection orbit", self.injection_orbit, "from probe files", self.tracking.name_dict)

    def print_bump(self, key, bump):
        print("Fore tracking bump with subs list:", key[0], "bump index", key[1])
        for magnet in bump["field_list"]:
            magnet_name = magnet.replace("_", " ")
            print(magnet_name, format(bump["field_list"][magnet], "10.6g"))
        print("Proton orbit:", bump["proton_orbit"])

    def get_field_list(self, a_bump):
        """
        Get the list of fields for a given bumper setting
        """
        return a_bump["bump_fields"]

    def get_proton_orbit(self, a_bump):
        """
        Get the target orbit at the foil for this bump setting
        """
        foil = str(self.config.track_bump["proton_orbit_station"])
        proton_orbit = a_bump["target_orbit"][foil]
        return proton_orbit

    def do_substitutions(self, fields, phi_init, charge):
        overrides = self.config.track_bump["subs_overrides"]
        #print json.dumps(self.subs, indent=2)
        n_cells = self.subs["__n_cells__"]
        if phi_init > n_cells/2:
            phi_init -= n_cells
        overrides["__beam_phi_init__"] = phi_init
        self.subs["__beam_charge__"] = charge
        overrides.update(fields)
        utilities.do_lattice(self.config, self.subs, overrides)

    def fore_track(self, fields, proton_orbit):
        self.file_index += 1
        foil_phi = self.config.track_bump["proton_orbit_phi"]
        foil_probe = self.config.track_bump["foil_probe_files"]
        energy = self.config.track_bump["energy"]
        self.do_substitutions(fields, foil_phi, +1.0)
        self.tracking = utilities.setup_tracking(self.config, foil_probe, energy)
        test_hit = utilities.reference(self.config, energy,
                                       proton_orbit[0], proton_orbit[1],
                                       proton_orbit[2], proton_orbit[3])
        try:
            track = self.tracking.track_many([test_hit])
        except OSError: # did not load any output files - we don't care
            track = None
        lattice_name = self.config.tracking["lattice_file_out"].split(".")[0]
        os.rename(lattice_name+"-trackOrbit.dat",
                  lattice_name+"-trackOrbit-fore-"+str(self.file_index)+".dat")
        return track

    def back_track(self, fields):
        self.file_index += 1
        foil_phi = self.config.track_bump["foil_phi"]
        injected_beam = self.injection_orbit
        foil_probe = self.config.track_bump["foil_probe_files"]
        energy = self.config.track_bump["energy"]
        self.do_substitutions(fields, foil_phi, +1.0)
        tracking = utilities.setup_tracking(self.config, foil_probe, energy)
        test_hit = utilities.reference(self.config, energy,
                                       injected_beam[0], injected_beam[1],
                                       injected_beam[2], injected_beam[3])
        test_hit["px"] *= -1
        test_hit["py"] *= -1
        test_hit["pz"] *= -1
        try:
            track = tracking.track_many([test_hit])
        except OSError: # did not load any output files - we don't care
            track = None
        lattice_name = self.config.tracking["lattice_file_out"].split(".")[0]
        os.rename(lattice_name+"-trackOrbit.dat",
                  lattice_name+"-trackOrbit-back-"+str(self.file_index)+".dat")
        return track

def track_painting(self, bump_list):
        foil_probe = self.config.track_bump["foil_probe_files"]
        ramp_probe = self.config.track_bump["ramp_probe_files"]
        foil_phi = self.config.track_bump["foil_probe_phi"]
        ramp_phi = self.config.track_bump["ramp_probe_phi"]
        injected_beam = self.injection_orbit
        energy = self.config.track_bump["energy"]

        self.do_substitutions({}, 0, +1.0)
        foil_tracking = utilities.setup_tracking(self.config, foil_probe, energy)
        ramp_tracking = utilities.setup_tracking(self.config, ramp_probe, energy)
        foil_tracking.verbose = False
        ramp_tracking.verbose = False
        test_hit = utilities.reference(self.config, energy,
                                        injected_beam[0], injected_beam[1],
                                        injected_beam[2], injected_beam[3])
        print("Seed kinetic energy:     ", test_hit["kinetic_energy"])
        # fix momentum
        hit_list_in = []
        log = open("track_bump.log", "w")
        for bump_fields, n_turns, closed_orbit in bump_list:
            # ramp the fields
            print("Ramping fields", bump_fields, file=log)
            if len(hit_list_in) > 0:
                # track from the ramp point to the foil point
                print("Tracking to foil", file=log)
                self.do_substitutions(bump_fields, ramp_phi, +1.0)
                hit_list_of_lists = foil_tracking.track_many(hit_list_in)
                hit_list_in = [hit_list[-1] for hit_list in hit_list_of_lists]
            self.do_substitutions(bump_fields, foil_phi, +1.0)
            for i in range(n_turns):
                hit_list_in.append(copy.deepcopy(test_hit))
                hit_list_in = [copy.deepcopy(test_hit), copy.deepcopy(test_hit)]+hit_list_in
                hit_list_of_lists = foil_tracking.track_many(hit_list_in)
                hit_list_of_lists = hit_list_of_lists[2:]
                hit_list_in = []
                for hit_list in hit_list_of_lists:
                    if hit_list[-1]["t"] > 100.:
                        hit_list_in.append(hit_list[-1])
                self.output.append(hit_list_in)
                print("  After turn", i+1, "fields", end=' ', file=log)
                for field in bump_fields:
                    print(field, end=' ')
                print("found", len(hit_list_of_lists), "particles", file=log)
                for item in hit_list_in:
                    print("   ", \
                          format(item["station"], "6"), \
                          format(item["t"], "12.6g"), \
                          format(item["x"], "12.6g"), \
                          format(item["px"], "12.6g"), \
                          format(item["kinetic_energy"], "12.6g"), file=log)
            # now track from the foil to the ramp point
            print("Tracking to ramp point", file=log)
            hit_list_of_lists = ramp_tracking.track_many(hit_list_in)
            hit_list_in = [hit_list[-1] for hit_list in hit_list_of_lists]
        return hit_list

def main(config):
    track_bump = TrackBump(config)
    track_bump.track_bumps()
    if __name__ == "__main__":
        input()
