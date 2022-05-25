import math
import json
import matplotlib
from optimisation_tools.utils import utilities
from optimisation_tools.toy_model._toy_model import ToyModel

def the_dir(my_dir):
    version = 2
    a_dir = my_dir+"/horizontal-injection-on-p_v"+str(version)
    return a_dir, version

def toy_models_single_turn_injection(foil_n_turns_list, my_dir):
    for foil, n_turns in foil_n_turns_list:
        t_list = [n for n in range(n_turns)]
        a_dir, version = the_dir(my_dir)
        output_dir = a_dir+"/n_"+str(n_turns)+"__foil_"+str(foil)+"/"
        study = a_dir.split("/")[-1].replace("_", " ")
        version = study.split(" v")[1]
        study = study.split(" v")[0]
        config = {
            "max_turn":n_turns+100,
            "turn_bumper_index":[t_list, t_list],
            "default_bumps":[["coupled", 40.0*i/(n_turns-1), 0.0, 0.0, 0.0] for i in t_list],
            "plot_frequency":10, #n_turns+1,
            "foil_column_density":foil,
            "foil_angle":90.0,
            "output_dir":output_dir,
            "number_per_pulse":1000,
            "closed_orbit":"output/baseline/closed_orbits_cache",
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
            "n_cells":15,
        }
        yield config
"""
def toy_models_correlated_painting(a_dir, angle_list):
    for angle_u, angle_v in angle_list:
        n_injection_turns = 10
        n_trajectory_turns = 10
        n_turns = n_injection_turns+n_trajectory_turns
        t_list = [n for n in range(n_turns)]
        output_dir = a_dir+"/test_u_"+str(math.degrees(angle_u))+"_v_"+str(math.degrees(angle_v))+"/"
        foil = 5e-6
        final_amplitude = 2.0
        bumps = [["action_angle", angle_u, i*final_amplitude/(n_turns-1),
                  angle_v, i*final_amplitude/(n_turns-1)] for i in range(n_turns)]
        config = {
            "max_turn":n_turns,
            "turn_bumper_index":[t_list, t_list],
            "default_bumps":bumps,
            "plot_frequency":n_turns+1,
            "foil_column_density":foil,
            "output_dir":output_dir,
            "number_pulses":n_injection_turns,
            "number_per_pulse":1000,
            "closed_orbit":"output/arctan_baseline/baseline/closed_orbits_cache",
            "injection_ellipse_algorithm":"from_twiss",
            "beta_x":2.0,
            "alpha_x":1.0,
            "beta_y":0.5,
            "alpha_y":-1.0,
        }
        yield config
"""

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
    for out_dict in output:
        for key, value in out_dict.items():
            #print(key, format(value, "6.4g"), end="  ") 
            print(key, value, end="  ") 
        print()
    a_dir, version = the_dir(my_dir)
    fout = open(a_dir+"/run_summary_list.json", "w")
    fout.write(json.dumps(output))

def main():
    a_dir = "output/baseline/toy_model/"
    output = []
    foil_n_turns_list = [(20e-6, n) for n in [50]]# (20e-6, 10)]
    toy_models = [toy for toy in toy_models_single_turn_injection(foil_n_turns_list, a_dir)]
    for i, config in enumerate(toy_models):
        out_dict = run_one(config, config != toy_models[-1])
        output.append(out_dict)
    save_output(output, a_dir)
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")

if __name__ == "__main__":
    main()

