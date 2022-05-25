
import json
import math
import os

def get_baseline_substitution(dipole_field, momentum, r0, wedge_angle, dp=[1, 1, 1, 1]):
    beamX, beamY = -13.114746147910342, -2.3088398137537665
    phi_s = 20
    efield = 43.82 # 21.2 @ phi_s 45 # 44.35 44.32 44.315 too high 44.3 44.31 too low
    sol_field = 9.0*4/5
    if r0 == 400.0:
        sol_field *= 1
    elif r0 == 300.0:
        sol_field *= (0.025/0.075)**0.5
    elif r0 == 200.0:
        sol_field *= (0.025/0.075)**0.5*(0.025/0.079)**0.5*1.1
    else:
        print("WARNING - not sure about sol field factor for r0", r0)
    baseline = {
        # ring
        "__cell_length__":2000, #mm
        "__dipole_field__":dipole_field, # T
        "__dipole_polarity1__":dp[0],
        "__dipole_polarity2__":dp[1],
        "__dipole_polarity3__":dp[2],
        "__dipole_polarity4__":dp[3],
        "__solenoid_field__":sol_field, # T
        "__solenoid_z_pos__":0.05, # fraction of a full cell length
        "__solenoid_polarity1__":+1,
        "__solenoid_polarity2__":-1,
        "__solenoid_polarity3__":+1,
        "__solenoid_polarity4__":-1,
        "__n_cells__":10,
        "__n_precells__":10,
        "__beam_x__":beamX,
        "__beam_xp__":0.0,
        "__beam_y__":beamY,
        "__beam_yp__":0.0,
        "__energy__":(momentum**2+105.658**2)**0.5-105.658, # kinetic, MeV
        "__momentum__":momentum, # MeV/c
        "__output_planes_per_cell__":1,
        "__output_cells__":3,
        "__max_step__":10,
        "__do_stochastics__":0,
        "__disable__":"Decay",
        "__do_cooling__":1,
        "__coil_radius__":r0,
        "__frequency__":0.704,
        "__efield__":efield, #26.03, # 30 MV/m at 704 MHz -> 26.03 MV/m at 352 MHz
        "__rf_phase__":phi_s,
        "__wedge_alignment_angle1__":-90,
        "__wedge_alignment_angle2__":90,
        "__wedge_opening_angle__":wedge_angle,
        "__wedge_thickness__":34.2,
        "__wedge_offset_z__":0.0,
        "__wedge_offset_trans__":beamX,
    }
    return baseline


#ISSUE: losing beam longitudinally. Not sure what is going on. Cannot recover longitudinal cooling performance from e.g. Stratakis simulation
#Try:
#1. Increase number of RF cavities to 6
#2. Increase bucket size by moving synch. phase towards 0 (also check I understand properly G4BL RF phase)
class Config(object):
    """
    """

    def __init__(self, dipole_field="0.0", momentum="200", r0="400", wedge_angle="000", dp_str="pppp"):
        delta = 1
        xp_ratio = 0.001
        by = float(dipole_field)
        pz = float(momentum)
        find_co = True
        track_beam = False
        name = "optics" #"cooling" # "tracking" # "optics"
        self.find_closed_orbits = {
            "seed":[[0.0, 0.0, 0.0, 0.0],],#0.2 T pmmp
            "fixed_variables":{},
            "deltas":[delta, delta*xp_ratio, delta, delta*xp_ratio],
            "adapt_deltas":False,
            "output_file":"closed_orbits_cache",
            "subs_overrides":{"__output_cells__":2, "__do_cooling__":0},
            "final_subs_overrides":{"__output_planes_per_cell__":100, "__output_cells__":2, "__do_cooling__":0, "__do_stochastics__":0},
            "plotting_subs":{},
            "us_cell":0,
            "ds_cell":1,
            "root_batch":0,
            "max_iterations":1,
            "tolerance":0.00001,
            "do_plot_orbit":False,
            "do_minuit":True,
            "minuit_weighting":{"x":1, "x'":0.0001, "y":1, "y'":0.0001},
            "minuit_tolerance":1e-10,
            "minuit_iterations":200,
            "run_dir":"tmp/find_closed_orbits",
            "probe_files":"output.txt",
            "overwrite":True,
            "orbit_file":"output.txt",
            "use_py_tracking":False,
            "py_tracking_phi_list":[0.0, math.pi*2.0/10.0],
        }
        self.find_da = {
            "run_dir":"tmp/find_da/",
            "probe_files":"output.txt",
            "subs_overrides":{"__output_cells__":30, "__n_cells__":40},
            "get_output_file":"get_da",
            "scan_output_file":"scan_da",
            "row_list":None,
            "scan_x_list":[],
            "scan_y_list":[],
            "x_seed":1.0,
            "y_seed":10.0,
            "min_delta":0.9,
            "max_delta":1000.,
            "required_n_hits":20,
            "dt_tolerance":0.5, # fraction of closed orbit dt
            "max_iterations":30,
            "decoupled":True,
            "save_dir":"find_da",
        }
        disp_x = (-16.6+9.9)/(210-190)*200
        disp_y = (-3.3+2.5)/(210-190)*200
        self.track_beam = {
            "run_dir":"tmp/track_beam/",
            "save_dir":"track_beam_amplitude",
            "print_events":[0, 1, -1],
            #"variables":["x", "px", "y", "py"],
            "settings":[{
                "name":"da_test",
                "direction":"forwards",
                "probe_files":"output.txt",          
                "beam":{
                    "type":"beam_gen",
                    "closed_orbit_file":"closed_orbits_cache",
                    "eigen_emittances":[[0, 0]]*2+[[25.0, 25.0]],
                    "n_per_dimension":20,
                    "variables":["x","x'","y","y'"],
                    "amplitude_dist":"uniform", #"grid", # 
                    "phase_dist":"uniform", #"grid", # 
                    "max_amplitude_4d":1000.0, # amplitude_dist != grid
                    "energy":(pz**2+105.658**2)**0.5-105.658,
                },
                "subs_overrides":{"__output_cells__":30, "__n_cells__":40, "__do_cooling__":0, "__do_stochastics__":0},
            },{
                "name":"equilibrium_test",
                "direction":"forwards",
                "probe_files":"output.txt",          
                "beam":{
                    "type":"grid",
                    "energy":(pz**2+105.658**2)**0.5-105.658,
                    "start":self.find_closed_orbits["seed"][0],
                    "stop":self.find_closed_orbits["seed"][0],
                    "nsteps":[1000, 1, 1, 1, 1, 1],
                    "reference":self.find_closed_orbits["seed"][0],
                },
                "subs_overrides":{"__output_cells__":100, "__n_cells__":110, "__do_cooling__":1, "__do_stochastics__":1},
            },{
                "name":"cooling_test",
                "direction":"forwards",
                "probe_files":"output.txt",          
                "beam":{
                    "type":"mvg",
                    "reference":self.find_closed_orbits["seed"][0]+[0.0, (pz**2+105.658**2)**0.5-105.658],
                    "closed_orbit_file":"closed_orbits_cache",
                    "eigenemittances":[2.5, 2.5],
                    #"cov_trans":[[1.0 if i==j else 0.0 for i in range(4)] for j in range(4)],
                    "cov_long":[[0.015, 0.0],[0.0, 20.0**2]],
                    "dispersion":[disp_x, 0.0, disp_y, 0.0], # dx, dxp, dy, dyp
                    "energy":(pz**2+105.658**2)**0.5-105.658,
                    "n_particles":1000,
                    "variables":["x","x'","y","y'"],
                    "variables_long":["t", "kinetic_energy"],
                },
                "subs_overrides":{"__output_cells__":100, "__n_cells__":105, "__do_cooling__":1, "__do_stochastics__":1},
            },][2:3],
        }
        """
        "x": -9.940249999999999, "y": -2.5414, "z": 0.0 @ p=190
        "x": -13.044, "y": -2.69238, "z": 0.0 @ p=200
        "x": -16.5891, "y": -3.28459, "z": 0.0 @ p=210
        """

        dp_list = [(+1 if s == "p" else -1 if s == "m" else 0) for s in dp_str]
        self.substitution_list = [get_baseline_substitution(float(by), float(pz), float(r0), float(wedge_angle), dp_list)]
        
        self.run_control = {
            "find_closed_orbits_4d":find_co,
            "find_tune":False,
            "find_da":False,
            "find_bump_parameters":False,
            "build_bump_surrogate_model":False,
            "track_bump":False,
            "track_beam":track_beam,
            "clean_output_dir":False,
            "output_dir":os.path.join(os.getcwd(), "output/rectilinear_cooling_v20/by="+str(dipole_field)+"_pz="+str(momentum)+"_r0="+str(r0)+"_wq="+wedge_angle+"_dp="+dp_str+"_"+name),
            "ffa_version":"2022-01-01 rectilinear",
            "root_verbose":6000,
            "faint_text":'\033[38;5;243m',
            "default_text":'\033[0m',
            "random_seed":0,
        }

        self.tracking = {
            "mpi_exe":None, #os.path.expandvars("${OPAL_ROOT_DIR}/external/install/bin/mpirun"),
            "beam_file_out":"beam.tmp",
            "n_cores":4,
            "links_folder":[], # link relative to lattice/VerticalFFA.in
            "lattice_file":os.path.join(os.getcwd(), "lattice/rectilinear_optimise/cooling.in"),
            "lattice_file_out":"cooling.g4bl",
            "opal_path":None,
            "g4bl_path":os.path.expandvars("${G4BL_EXE_PATH}/g4bl"), # only used of opal path is None
            "g4bl_beam_format":"g4beamline_bl_track_file",
            "g4bl_output_format":"icool_for009",
            "tracking_log":"log",
            "flags":[],
            "step_size":1.,
            "ignore_events":[],
            "pdg_pid":-13,
            "clear_files":"*.h5",
            "verbose":0,
            "file_format":"hdf5",
            "analysis_coordinate_system":"none",
            "dt_tolerance":-1., # ns
            "station_dt_tolerance":-1., # ns, if +ve and two hits are close together, reallocate station
            "py_tracking":{
                "derivative_function":"u_derivative",
                "atol":1e-12,
                "rtol":1e-12,
                "verbose":True,
            }
        }

