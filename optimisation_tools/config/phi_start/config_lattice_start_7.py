import bisect
import os
import config.config_lattice_start as config
import plotting.plot_orbit as plot_orbit

ANGLE = 7

def get_seed(angle):
    filename = "output/jan_baseline/tmp/find_closed_orbits/VerticalFFAGMagnet-trackOrbit_1.dat"
    step_list = plot_orbit.parse_track_file(filename)
    while angle < 0:
        angle += 360.
    while angle > 360:
        angle -= 360
    delta = [abs(angle-item) for item in step_list['phi']]
    index = delta.index(min(delta))
    print("Closest angle to", angle, "was", step_list['phi'][index], "from range",
          min(step_list['phi']), "to", max(step_list['phi']))
    units = {'r':1e3, 'pr':1, 'z':1e3, 'pz':1}
    seed = [step_list[key][index]*units[key] for key in ['r', 'pr', 'z', 'pz']]
    print("Found element", end=" ")
    for key in sorted(step_list.keys()):
        print(key, step_list[key][index], end = " ")
    print("\n    Found seed for angle", angle, ":", seed)
    return seed

class Config(config.Config):
    def __init__(self):
        super(Config, self).__init__()
        angle = ANGLE
        self.substitution_list = [config.get_baseline_substitution()]
        self.substitution_list[-1]["__lattice_phi_start__"] = angle
        self.run_control["output_dir"] = os.path.join(os.getcwd(),
                                                      "output/jan/phi_start/"+str(angle))
        self.find_closed_orbits["seed"] = [
            get_seed(angle+90)
        ]

