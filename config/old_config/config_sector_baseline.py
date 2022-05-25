import math
import os

def get_baseline_substitution():
    baseline = {
        # ring
        "__n_cells__":10,
        "__radius__":3.995285, #m
        "__half_cell_length__":1.25,
        "__n_particles__":3,
        # beam
        "__energy__":3., # MeV
        "__beam_phi_init__":0., # fraction of cell length
        # tracking
        "__step_size__":0.001, # m
        "__n_turns__":0.2,
        # main magnets
        "__n_sectors__":7,
        "__occupancy__":0.8,
        "__bf__":-0.5, #-0.25/0.44, # T
        "__bd__":0.22, #0.25, #+0.20, # T
        "__d_length__":0.5, # m
        "__f_length__":0.5, # m
        "__d_offset__":-0.162, # m # 0.154
        "__f_offset__":0.162, # m
        "__d_end_length__":0.15,  # m
        "__f_end_length__":0.15, # m
        "__m_index__":1.44, # m^-1 # 1.58
        "__max_x_power__":10,
        "__neg_extent__":2.0, # m
        "__pos_extent__":4.0, # m
        "__f_bend_angle__":+32.0,
        "__d_bend_angle__":-16.0,
        # NOTE BB length 0.125 m -> width 1.0 m; length 2.0 m; double for safety
        "__magnet_width__":4.0, # m
        "__bb_length__":8.0, # m
        # field maps
        "__do_magnet_field_maps__":True,
        "__cartesian_x_min__":-5., # m
        "__cartesian_dx__":0.01, # m (1001 steps)
        "__cartesian_y_min__":-5., # m
        "__cartesian_dy__":0.01, # m (1001 steps)
        # bump magnets
        "__do_bump__":False,
        "__foil_probe_phi__":2.1,
        "__bump_4_probe_phi__":2.55,
        "__h_bump_1_field__":0.0,
        "__h_bump_2_field__":0.0,
        "__h_bump_3_field__":0.0,
        "__h_bump_4_field__":0.0,
        "__h_bump_5_field__":0.0,
        "__v_bump_1_field__":0.0,
        "__v_bump_2_field__":0.0,
        "__v_bump_3_field__":0.0,
        "__v_bump_4_field__":0.0,
        "__v_bump_5_field__":0.0,
        "__h_bump_1_phi__":1.43, #1.43,
        "__h_bump_2_phi__":1.93, #2.43,
        "__h_bump_3_phi__":2.07, #3.43,
        "__h_bump_4_phi__":2.43, #4.43,
        "__h_bump_5_phi__":2.57, #4.43,
        "__v_bump_1_phi__":1.43,
        "__v_bump_2_phi__":1.93,
        "__v_bump_3_phi__":2.07,
        "__v_bump_4_phi__":2.43,
        "__v_bump_5_phi__":2.57,
        "__h_bump_1_dphi__":-0.25,
        "__h_bump_2_dphi__":+0.25,
        "__h_bump_3_dphi__":+0.25,
        "__h_bump_4_dphi__":-0.25,
        "__h_bump_5_dphi__":-0.25,
        "__v_bump_1_dphi__":-0.25,
        "__v_bump_2_dphi__":+0.25,
        "__v_bump_3_dphi__":+0.25,
        "__v_bump_4_dphi__":-0.25,
        "__v_bump_5_dphi__":-0.25,
        "__septum_field__":0.0, #-0.20832864219,
        "__septum_length__":0.15, # m
        "__septum_fringe__":0.1, # fraction of septum_length
        "__septum_width__":0.5, # m
        "__septum_phi__":0.5, # fraction of cell length
        "__septum_dr__":0.5, # m
    }
    return baseline

class Config(object):
    def __init__(self):
        self.find_closed_orbits = { 
                #order 6 [3944.5849617875692, -26.049083417439988, 158.4519347916439, -1.5875088457067648]
                #order 10 [3971.0069543538857, -22.02559197601977, 74.96704487122588, -1.2762095154819235]],
            "seed":[[3971.0069543538857, -22.02559197601977, 74.96704487122588, -1.2762095154819235]],
            "deltas":[1.0]*4,
            "adapt_deltas":False,
            "output_file":"closed_orbits_cache",
            "subs_overrides":{"__n_turns__":0.21, "__do_magnet_field_maps__":"False"},
            "final_subs_overrides":{"__n_turns__":1.21, "__do_magnet_field_maps__":"True"},
            "us_cell":0,
            "ds_cell":1,
            "root_batch":0,
            "max_iterations":5, 
            "tolerance":0.1,
            "do_plot_orbit":False,
            "run_dir":"tmp/find_closed_orbits",
            "probe_files":"RINGPROBE*.h5",
            "overwrite":True,
            "orbit_file":"VerticalSectorFFA-trackOrbit.dat",
        }
        self.find_tune = {
            "run_dir":"tmp/find_tune/",
            "probe_files":"RINGPROBE01.h5",
            "subs_overrides":{"__n_turns__":120.1, "__do_magnet_field_maps__":"True"},
            "root_batch":0,
            "delta_1":5.,
            "delta_2":5.,
            "max_n_cells":0.1,
            "output_file":"find_tune",
            "row_list":None,
            "axis":None,
        }
        self.find_da = {
            "run_dir":"tmp/find_da/",
            "probe_files":"RINGPROBE01.h5",
            "subs_overrides":{"__n_turns__":120.1, "__do_magnet_field_maps__":"False", "__step_size__":0.001},
            "get_output_file":"get_da",
            "scan_output_file":"scan_da",
            "row_list":None,
            "scan_x_list":[],
            "scan_y_list":[],
            "x_seed":32.,
            "y_seed":32.,
            "min_delta":0.9,
            "max_delta":1000.,
            "required_n_hits":100,
            "dt_tolerance":0.5, # fraction of closed orbit dt
            "max_iterations":15,
            "decoupled":True,
        }
        co = self.find_closed_orbits["seed"][0]
        bump = []
        for xi in [0, 1, 2, -2, -1]:
            for yi in range(5):
                bump.append((0, [10.0*xi, 0.0, -10.0*yi, 0.0]))
        self.find_bump_parameters = {
            "n_h_bumps":5,
            "n_v_bumps":5,
            "amp":[],
            "output_file":"find_bump_parameters",
            "closed_orbit":co,
            "magnet_min_field":-1.0,
            "magnet_max_field":+1.0,
            "max_iterations":5000,
            "field_tolerance":1e-4,
            "amplitude_tolerance":1.,
            "tm_source":"closed_orbits_cache",
            "subs_overrides":{
                "__n_turns__":1.1,
                "__do_magnet_field_maps__":False,
                "__max_x_power__":10,
                "__do_bump__":True,
            },
            "final_subs_overrides":{
                "__n_turns__":1.1,
                "__max_x_power__":10,
                "__do_magnet_field_maps__":True,
                "__do_bump__":True,
            },
            "bump":bump,
            "staged_optimisation":[{ # get foil position
                    "seed_fields":{
                                "__h_bump_1_field__":0.0,
                                "__v_bump_1_field__":0.0,
                                "__h_bump_2_field__":0.0,
                                "__v_bump_2_field__":0.0,
                                "__h_bump_3_field__":0.0,
                                "__v_bump_3_field__":0.0,
                                "__h_bump_4_field__":0.0,
                                "__v_bump_4_field__":0.0,
                                "__h_bump_5_field__":0.0,
                                "__v_bump_5_field__":0.0,
                    },
                    "target_orbit":dict([
                        (0, [3897.68994, -17.78, 89.38, 3.927]),
                        ]+[(i, co) for i in (1, 2,)]),
                    "position_tolerance":0.01,
                    "momentum_tolerance":0.001,
                    "fix_bumps":["__v_bump_-1_field__", "__h_bump_-1_field__",
                                 "__v_bump_-2_field__", "__h_bump_-2_field__",
                                 "__v_bump_3_field__", "__h_bump_3_field__",
                                 "__v_bump_4_field__", "__h_bump_4_field__",
                                 "__v_bump_5_field__", "__h_bump_5_field__",],
                }, { # recover closed orbit
                    "seed_fields":{},
                    "target_orbit":dict([ #maps station:target orbit
                        ]+[(i, co) for i in (1, 2, 4)]),
                    "position_tolerance":0.01,
                    "momentum_tolerance":0.001,
                    "fix_bumps":["__v_bump_1_field__", "__h_bump_1_field__",
                                 "__v_bump_2_field__", "__h_bump_2_field__",
                                 "__v_bump_3_field__", "__h_bump_3_field__",
                                 "__v_bump_-4_field__", "__h_bump_-4_field__",
                                 "__v_bump_-5_field__", "__h_bump_-5_field__",],
                }#
            ],
            "target_fields":{},
            "seed_errors":[1e-3]*10,
            "ref_probe_files":["FOILPROBE.h5", "RINGPROBE*.h5"], # sorted alphanumerically
            "run_dir":"tmp/find_bump/",
            "energy":3.0,
            "min_time_delta":10., # minimum time between probes
            "target_n_hits":3,
            "penalty_factor":1e9, # penalty = p_f^(number of missed stations)
            "algorithm":"migrad",
        }
        self.track_bump = {
            "input_file":"find_bump_parameters.tmp",
            "foil_optimisation_stage":0, # from staged optimisation
            "field_optimisation_stage":1, # from staged optimisation
            "proton_orbit_station":0, # proton orbit taken from this station
            "proton_orbit_phi":0., # protons start here
            "subs_overrides":{
                "__n_turns__":1.2,
                "__do_magnet_field_maps__":True,
                "__do_bump__":True,
            },
            "injection_orbit":[3897.68994, -17.78, 89.38, 3.927],
            "bump_list":None, # list of bumps which we will track (default to find_bump_parameters)
            "n_turns_list":None, # number of turns for forward tracking (default 1)
            "foil_probe_files":["FOILPROBE.h5"], # PROBE where the beam is injected
            "ramp_probe_files":["RINGPROBE06.h5"], # PROBE where the magnet is ramped
            "ramp_probe_phi":5, # cell number where we start the beam following an injection
            "run_dir":"tmp/track_bump/",
            "energy":3.0,
        }

        self.substitution_list = [get_baseline_substitution()]
        
        self.run_control = {
            "find_closed_orbits_4d":True,
            "find_tune":False,
            "find_da":True,
            "find_bump_parameters":False,
            "track_bump":False,
            "clean_output_dir":False,
            "output_dir":os.path.join(os.getcwd(), "output/sector_baseline/baseline"),
            "root_verbose":6000,
            "faint_text":'\033[38;5;243m', #253m
            "default_text":'\033[0m'
        }

        self.tracking = {
            "mpi_exe":None, #os.path.expandvars("${OPAL_ROOT_DIR}/external/install/bin/mpirun"),
            "beam_file_out":"disttest.dat",
            "n_cores":4,
            "links_folder":["VerticalSectorFFA",], # link relative to lattice/VerticalFFA.in
            "lattice_file":os.path.join(os.getcwd(), "lattice/VerticalSectorFFA.in"),
            "lattice_file_out":"VerticalSectorFFA.tmp",
            "opal_path":os.path.expandvars("${OPAL_EXE_PATH}/opal"),
            "tracking_log":"log",
            "flags":[],
            "step_size":1.,
            "ignore_events":[0],
            "pdg_pid":2212,
            "clear_files":"*.h5",
            "verbose":0,
            "file_format":"hdf5",
            "analysis_coordinate_system":"azimuthal",
            "dt_tolerance":100., # ns
        }

