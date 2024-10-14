import math
import json
import matplotlib
from optimisation_tools.utils import utilities
from optimisation_tools.toy_model._toy_model import ToyModel

def the_dir(my_dir):
    version = 1
    a_dir = my_dir+"/horizontal-injection-on-p_v"+str(version)
    return a_dir, version

def toy_models_single_turn_injection(foil_n_turns_list, my_dir):
    for foil, n_turns in foil_n_turns_list:
        a_dir, version = the_dir(my_dir)
        output_dir = a_dir+"/n_"+str(n_turns)+"__foil_"+str(foil)+"/"
        study = a_dir.split("/")[-1].replace("_", " ")
        version = study.split(" v")[1]
        study = study.split(" v")[0]
        config = {
            "max_turn":n_turns+100,
            "calculated_bumps":[["coupled", 40.0*i/(n_turns-1), 0.0, 0.0, 0.0] for i in range(n_turns)],
            "plot_frequency":1, #n_turns+1,
            "foil_column_density":foil,
            "foil_angle":90.0,
            "output_dir":output_dir,
            "number_per_pulse":1000,
            "closed_orbit":"output/2023-03-01_baseline/baseline/closed_orbits_cache",
            "injection_ellipse_algorithm":"from_twiss",
            "m_index":7.1,
            "harmonic_number":2.0,
            "beta_x":2.0,
            "alpha_x":0.0,
            "beta_y":2.0,
            "alpha_y":0.0,
            "beam_pulses":[[0.5-175.0/1149, 0.5+175.0/1149]], #[[-0.33/2, +0.33/2], [0.5-0.33/2, 0.5+0.33/2]], 
            "dp_over_p":0.0013,
            "dp_model":"gauss",#"none", #
            "do_movie":True,
            "pulse_emittance":0.026*1e-3,
            "lattice":"hffa fets ring v1",
            "study":study,
            "version":version,
            "seed":4,
            "n_foil_sigma":3,
            "foil_de_for_rf_bucket":0.0,
            "momentum":75.0,
            "rf_reference_momentum":75.0,
            "sleep_time":-1,
            "rf_voltage":0.004,
            "n_cells":1,
        }
        yield config

def toy_models_painting(angle_u, angle_v, a_dir, is_correlated, n_injection_turns, version, config_update):
    if is_correlated:
        study = "correlated_painting"
    else:
        study = "anticorrelated_painting"
    n_trajectory_turns = 10
    n_turns = n_injection_turns+n_trajectory_turns
    output_dir = a_dir+f"/test_toy_models_{study}_v{version}/"
    foil = 20e-6
    injection_amplitude = 0.01
    if is_correlated:
        injection_scale = [i**0.5/(n_injection_turns-1)**0.5 for i in range(n_injection_turns+1)]
    else:
        injection_scale = [i/(n_injection_turns-1) for i in range(n_injection_turns+1)]

    if is_correlated:
        inj   = [["action_angle", 0.0, 0.0, angle_v, scale*injection_amplitude] for scale in injection_scale]
    else:
        inj   = [["action_angle", 0.0, 0.0, angle_v, scale*injection_amplitude] for scale in reversed(injection_scale)]

    bumps = [["action_angle", angle_u, scale*injection_amplitude, 0.0, 0.0] for scale in injection_scale]
    bumps += [["delta", 20.0/n_trajectory_turns, 0.05/n_trajectory_turns, 0.0, 0.0] for i in range(n_trajectory_turns)]
    config = {
        "verbose":1,
        "max_turn":n_turns,
        "calculated_bumps":bumps,
        "calculated_injection":inj,
        "plot_frequency":1,
        "foil_column_density":foil,
        "output_dir":output_dir,
        "number_pulses":n_injection_turns,
        "number_per_pulse":int(100000/n_injection_turns),
        "closed_orbit":"output/2023-03-01_baseline/baseline/closed_orbits_cache",
        "injection_ellipse_algorithm":"transfer_matrix",
        "beta_x":2.0,
        "alpha_x":0.0,
        "beta_y":2.0,
        "alpha_y":0.0,
        "rf_voltage":0.006,
        "foil_angle":90.0,
        "n_cells":1,
        "pulse_emittance":0.026*1e-3,
        "tof_multiplier":16*1.0172,
        "harmonic_number":2,
        "amplitude_acceptance":0.020,
        "do_movie":True,
        "lattice":"2023-03-01 baseline",
        "study":study.replace("_", " "),
        "version":version,
        "m_index":7.4561,
        "verbose_particles":[],
        "sleep_time":-1, # set to positive time to print screens
    }
    config.update(config_update)
    return config

def run_one(config, will_clear):
    model = ToyModel()
    #model.verbose = -1
    model.parse_args()
    if config != None:
        model.do_config(config)
    model.initialise()
    model.run_toy_model()
    params = model.get_output_parameters()
    model.finalise(will_clear)
    return params

def save_output(output, my_dir):
    if False:
        for out_dict in output:
            for key, value in out_dict.items():
                #print(key, format(value, "6.4g"), end="  ")
                print(key, value, end="  ")
            print()
    #a_dir, version = the_dir(my_dir)
    fout = open(my_dir+"/run_summary_list.json", "w")
    fout.write(json.dumps(output, indent=2))

def main():
    output = []
    #a_dir = "output/2023-03-01_baseline/baseline/toy_model_single_turn/"
    #foil_n_turns_list = [(20e-6, n) for n in [50]]
    #toy_models = [toy for toy in toy_models_single_turn_injection(foil_n_turns_list, a_dir)]

    a_dir = "output/2023-03-01_baseline/toy_model_painting_v20/"
    config_list = []
    for is_correlated in [False, True]:
        version = 1
        for rf_voltage_i in range(0, 1, 2):
            rf_voltage = rf_voltage_i*1e-3
            for amplitude_acceptance_i in range(4, 5, 4):
                amplitude_acceptance = amplitude_acceptance_i*0.005
                for dp_i in range(0, 1, 2):
                    dp = (1+dp_i*1e-3)*75
                    for n_injection_turns in range(50, 101, 100):
                        version = n_injection_turns
                        config = {
                            "angle_u":math.pi/2,
                            "angle_v":0*math.pi/2*0.92,
                            "a_dir":a_dir,
                            "is_correlated":is_correlated,
                            "n_injection_turns":n_injection_turns,
                            "version":version,
                            "config_update":{
                                "rf_reference_momentum":dp,
                                "rf_voltage":rf_voltage,
                                "amplitude_acceptance":amplitude_acceptance,
                                "output_dir":a_dir+f"/corr={is_correlated}_dp={dp}_n-turns={n_injection_turns}/",
                                "plot_frequency":1e9,
                                "do_movie":False
                            },
                        }
                        config_list.append(config)

    for i, config in enumerate(config_list):
        toy_model = toy_models_painting(**config)
        out_dict = run_one(toy_model, i != len(config_list)-1)
        config.update(toy_model)
        out_dict["config"] = config
        output.append(out_dict)
        if i < 3:
            save_output(output, a_dir)
    save_output(output, a_dir)
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")

if __name__ == "__main__":
    main()

