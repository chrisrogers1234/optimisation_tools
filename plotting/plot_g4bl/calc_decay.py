import math
import xboa.common

def calc_decay(distance_m, pz_MeVc):
    pz = pz_MeVc
    distance = distance_m*1000
    lifetime = 2200
    mass = xboa.common.pdg_pid_to_mass[13]
    c_light = xboa.common.constants["c_light"]
    energy = (pz**2+mass**2)**0.5
    gamma = energy/mass
    velocity = pz/energy*c_light # mm/ns
    time = distance/velocity
    decay = math.exp(-time/lifetime/gamma)
    return decay


def main():
    distance = 100 # metres
    pz = 300 # MeV/c
    print("Survival rate for pz", pz, "MeV/c and distance", distance, "m is", calc_decay(distance, pz))

if __name__ == "__main__":
    main()
