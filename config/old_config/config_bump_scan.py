import math
import os
from . import config_triplet_baseline as config

def get_baseline_substitution():
    sub = config.get_baseline_substitution()
    sub["__h_bump_1_phi__"] = 1.00
    sub["__h_bump_2_phi__"] = 2.00
    sub["__h_bump_3_phi__"] = 3.07
    sub["__h_bump_4_phi__"] = 4.00
    sub["__h_bump_5_phi__"] = 5.00
    sub["__v_bump_1_phi__"] = 1.00
    sub["__v_bump_2_phi__"] = 2.00
    sub["__v_bump_3_phi__"] = 3.07
    sub["__v_bump_4_phi__"] = 4.00
    sub["__v_bump_5_phi__"] = 5.00
    sub["__h_bump_1_dphi__"] = 0.0
    sub["__h_bump_2_dphi__"] = 0.0
    sub["__h_bump_3_dphi__"] = 0.25
    sub["__h_bump_4_dphi__"] = 0.0
    sub["__h_bump_5_dphi__"] = 0.0
    sub["__v_bump_1_dphi__"] = 0.0
    sub["__v_bump_2_dphi__"] = 0.0
    sub["__v_bump_3_dphi__"] = 0.25
    sub["__v_bump_4_dphi__"] = 0.0
    sub["__v_bump_5_dphi__"] = 0.0
    sub["__foil_probe_phi__"] = 3.095
    sub["__foil_probe_dphi__"] = 0.25/2.+0.025*math.pi/10.
    return sub

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.substitution_list = []
        bump_fields = {
            "__h_bump_1_field__" : [-4.853524556036604e-06, -0.0009517765658660915, -0.0019889314356771326, -0.0031518729689510305, -0.004429527643582176, -0.005812086673142813, -0.00728065701847469, -0.008954545703522054, -0.010682797212251183, -0.01259209857408583, -0.0146210430036674] ,
            "__v_bump_1_field__" : [1.3174889596845318e-05, 0.0004674425803554705, 0.0009320906028014164, 0.0013303269192090905, 0.001722780248733402, 0.0020475607449970123, 0.002353903508182986, 0.0025778456593010812, 0.002775762388822267, 0.0029560664893857336, 0.003085036234049099] ,
            "__h_bump_2_field__" : [3.2605729495926994e-05, -0.003514188463293988, -0.007092645007654941, -0.010729677240006397, -0.014359957958728575, -0.0181124125097869, -0.021887552278833322, -0.02562447863576467, -0.029450225480245118, -0.03330117428531032, -0.03721453135059882] ,
            "__v_bump_2_field__" : [-1.4354124264981394e-05, -0.0006924213463102014, -0.0013994974613454891, -0.002035146446714986, -0.0026708169592293274, -0.0032274100440543574, -0.003775975844161117, -0.004228626764441246, -0.00465317877023852, -0.005058144215815519, -0.005396053605674189] ,
            "__h_bump_3_field__" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            "__v_bump_3_field__" : [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] ,
            "__h_bump_4_field__" : [0.0026136496770530204, -0.0002663778395773919, -0.00310292584756533, -0.005952466280393698, -0.008822577601959036, -0.011534988461770146, -0.014165575580694978, -0.01681902702023974, -0.019378078971239554, -0.02178082891529054, -0.024073621909673437] ,
            "__v_bump_4_field__" : [-0.05994471454564687, -0.06015386389317667, -0.060278385249222466, -0.060311555799095085, -0.06027167682381618, -0.06014094470971221, -0.059927607987541265, -0.05962962315390696, -0.05922287138892246, -0.058632927275429014, -0.057910984908783214] ,
            "__h_bump_5_field__" : [0.004986952958933477, 0.0014247892612717639, -0.0023801207682576653, -0.006312581768236458, -0.010304078788099025, -0.014712225935858969, -0.019366415889909838, -0.024069012847677707, -0.029072730767268573, -0.03442421918855576, -0.04011858427786108] ,
            "__v_bump_5_field__" : [0.04922569223549966, 0.04871055316486994, 0.04811444002046872, 0.047494445474800084, 0.04686609056940805, 0.04618972599867277, 0.04546890199239195, 0.04478306962798584, 0.044070944393069134, 0.04330527306049192, 0.04253917431913656] ,
        }
        nsteps = len(bump_fields["__h_bump_1_field__"])
        for i in range(nsteps):
            sub = get_baseline_substitution()
            for key in bump_fields:
                sub[key] = bump_fields[key][i]
            self.substitution_list.append(sub)
            self.run_control["find_closed_orbits_4d"] = False
            self.run_control["find_tune"] = False
            self.run_control["find_da"] = True
            self.run_control["find_bump_parameters"] = False
            self.find_closed_orbits["deltas"] = [0.1, 0.1, 0.1, 0.1]
            self.find_closed_orbits["subs_overrides"]["__step_size__"] = 0.0001
            self.find_closed_orbits["subs_overrides"]["__do_bump__"] = True
            self.find_closed_orbits["subs_overrides"]["__n_turns__"] = 2.21
            self.find_closed_orbits["final_subs_overrides"]["__step_size__"] = 0.0001
            self.find_closed_orbits["final_subs_overrides"]["__n_turns__"] = 2.21
            self.find_closed_orbits["final_subs_overrides"]["__do_bump__"] = True
            self.find_closed_orbits["us_cell"] = 1
            self.find_closed_orbits["ds_cell"] = 2
            self.find_closed_orbits["probe_files"] = "RINGPROBE01.h5"
            self.find_closed_orbits["max_iterations"] = 5
            self.find_tune["delta_1"] = 5.
            self.find_tune["delta_2"] = 5.


        self.run_control["output_dir"] = os.path.join(os.getcwd(), "output/triplet_baseline/track_bump-8")
