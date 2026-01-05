"""
Physical constants
"""

def get_mass(pid):
    """Return the mass in MeV/c^2 for a given pid"""
    return pid_to_mass[abs(pid)]

def get_name(pid):
    """Return a human-readable name for a given pid"""
    for key, value in name_to_pid.items():
        if value == name:
            return key
    raise ValueError("Did not recognise pid "+str(pid))

def get_pid(name):
    """Return a pid from a given name"""
    return name_to_pid[name]
  
def get_charge(pid):
    """Return the charge, in units of positron charge, for a given pid"""
    return pid_to_charge[pid]

units = {
    "u":931.4940954, # MeV/c2
    "g":5.60958865e26, # MeV/c2
}

c_light = 29.9792458 # cm/ns
fine_structure_constant = 1./137.035999173 # dimensionless
bohr_radius = 5.291772106e-9 # cm
cross_section_norm = 1e-16 # cm^2
rydberg_energy = 13.605693009e-6 # MeV
avogadro_constant = 6.02214076e23

# 99LZZZAAAEEEI denote ions where 
# EEE enumerates the number of electrons
# ZZZ enumerates the number of protons
# AAA enumerates the number of nucleons
# I is the isomer
name_to_pid = {
    "proton":2212,
    "e+":-11,
    "mu+":-13,
    "e-":11,
    "mu-":13,
    "H-":9900010010020,
    "H":9900010010010,
    "H+":9900010010000,
}

# units are MeV/c^2
pid_to_mass = {
    2212:938.2720813,
    13:105.6583745,
    11:0.5109989461,
    9900010010020:1.0085*units["u"],
    9900010010010:1.0080*units["u"],
    9900010010000:1.0075*units["u"],
}

pid_to_charge = {
    2212:1.,
    -11:+1.,
    +11:-1.,
    -13:+1.,
    +13:-1.,
    9900010010020:-1,
    9900010010010:0,
    9900010010000:1,
}
