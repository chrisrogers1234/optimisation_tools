import json
import math
import copy
import os

import numpy.polynomial # hermite polynomials
import xboa.common

from . import particle
from . import constants

class Material(object):
    def __init__(self):
        self.density = 0.
        self.ion_energy = 0.
        self.molecular_mass = 0.
        self.elements_list = []
        self.X0 = 0.
        self.stripping_algorithm = "saha"
        self.straggling_hermite = None

    def set_material(self, material_name, density_factor=1):
        fin = open(self.materials_file)
        materials = fin.read()
        materials = json.loads(materials)
        my_mat = materials[material_name]
        self.name = material_name
        self.density = my_mat["density"]*density_factor
        self.ion_energy = my_mat["ion_energy"]*1e-6
        self.delta = my_mat["delta"]
        self.molecular_mass = my_mat["molecular_mass"] #*constants.units["u"]
        self.elements_list = my_mat["elements_list"]
        self.X0 = my_mat["radiation_length"]/self.density # cm
        # E_t has units of MeV; other constants are dimensionless
        self.stripping_n10_Et = my_mat["sigma_0,1"]["E_t"]
        a_list = ["a_"+str(i) for i in range(1,7)]
        self.stripping_0p1_Et = my_mat["sigma_0,1"]["E_t"]
        self.stripping_n10_Et = my_mat["sigma_-1,0"]["E_t"]
        # 0th element is None to be consistent with paper (which indexes from 1)
        self.stripping_0p1_a = [None]+[my_mat["sigma_0,1"][a_i] for a_i in a_list]
        self.stripping_n10_a = [None]+[my_mat["sigma_-1,0"][a_i] for a_i in a_list]
        self.heat_capacity = my_mat["specific_heat"] # J/g K

    def energy_loss_dz(self, particle):
        """
        PDG model for bethe-bloch energy loss
        """
        # bethe model for beta gamma > 0.05 (proton energy > 1 MeV)
        bg2 = particle.beta_rel**2*particle.gamma_rel**2
        m_ratio = self.m_e/particle.mass
        t_max = 2.0*self.m_e*bg2/(1.0 + 2.0*particle.gamma_rel*m_ratio+m_ratio**2)
        coefficient = math.log(2*self.m_e*bg2*t_max/(self.ion_energy**2))/2.
        coefficient -= particle.beta_rel**2 # - self.delta/2.
        coefficient *= 0.307075*particle.charge**2/particle.beta_rel**2*self.density

        dEdz = 0.
        for element in self.elements_list:
            dEdz += coefficient * element["z"]/element["a"]*element["fraction"]
        return -dEdz

    def energy_straggling_moments(self, particle, thickness):
        """
        Calculate the energy straggling moments for an incident particle, following
        Rotondi and Montagna NIM B 47 (1990).
        """
        m_e = xboa.common.pdg_pid_to_mass[11] # electron mass
        eps_max = 2*m_e*particle.beta_rel**2*particle.gamma_rel**2/(1+2*particle.gamma_rel*m_e/particle.mass)
        eta_const = 0.1536*particle.charge**2/particle.beta_rel**2*thickness
        eta = 0.0
        for element in self.elements_list:
            eta += eta_const*element["z"]/element["a"]*element["fraction"]
        kappa = eta/eps_max
        #print("kappa", kappa, "eta", eta, "eps_max", eps_max)
        alpha = [0.0 for i in range(6)]
        for n in range(2, 6):
            alpha[n] = eta**n/kappa**(n-1)*(1/(n-1) - particle.beta_rel**2/n)
        mu = [1.0, 0.0, alpha[2], alpha[3], 3*alpha[2]**2+alpha[4], 10*alpha[2]*alpha[3]+alpha[5]]
        return mu

    def set_energy_straggling_distribution(self, particle, thickness):
        """
        Calculate the energy straggling for an incident particle, following
        Rotondi and Montagna NIM B 47 (1990).
        """
        mu = self.energy_straggling_moments(particle, thickness)
        sigma = mu[2]**0.5
        mu_sig = [0.0, 0.0, 1.0]+[mu[i]/mu[2]**(i/2.0) for i in range(3, 6)] # mu[i]/sigma^i
        fac = [math.factorial(i) for i in range(10)]
        hermite_coeffs = [1, 0.0, 0.0,
            1/fac[3]*mu_sig[3],
            1/fac[4]*(mu_sig[4]-3),
            1/fac[5]*(mu_sig[5]-10*mu_sig[3]),
            10/fac[6]*mu_sig[3]**2,
            35/fac[7]*mu_sig[3]*(mu_sig[4]-3),
            280/fac[9]*mu_sig[3]**3
        ]
        self.straggling_hermite = numpy.polynomial.Hermite(hermite_coeffs)
        return self.straggling_hermite

    def energy_straggling(self, particle, thickness, delta):
        mu = self.energy_straggling_moments(particle, thickness)
        herm = self.set_energy_straggling_distribution(particle, thickness)
        return herm(delta)*1./(2*math.pi*mu[2])**0.5 * math.exp(- delta*delta/2/mu[2])

    def stripping_cross_section_nakai(self, particle):
        """
        Calculate the cross section for incident particle
        - particle: uses the energy and particle type (number of electrons) to
          determine which cross section formula to use, then applies the formula
          of nakai et al
        """
        # Y. Nakai et al., Charge Transfer of Hydrogen Atoms and Ions, Atomic
        # Data and Nuclear Data Tables 37, 69-101 (1987)
        if particle.pid == 9900010010020:
            e_t = self.stripping_n10_Et
            a = self.stripping_n10_a
            stripped = 9900010010010
        elif particle.pid == 9900010010010:
            e_t = self.stripping_0p1_Et
            a = self.stripping_0p1_a
            stripped = 9900010010000
        else:
            raise ValueError("Stripping cross sections can only be calculated "+
                             "for incident Hydrogen ions")
        e_1 = (particle.get_kinetic_energy() - e_t)*1e6 # eV
        if e_1 < 0:
            return 0.
        e_r = self.e_r*1e6 # eV
        f_e_top = a[1]*(e_1/e_r)**a[2]
        # a[3] and a[5] have units of keV
        f_e_bottom = (1+(e_1/a[3]/1e3)**(a[2]+a[4])+(e_1/a[5]/1e3)**(a[2]+a[6]))
        cross_section = constants.cross_section_norm*f_e_top/f_e_bottom
        return cross_section

    def stripping_cross_section_saha(self, a_particle):
        """
        Calculate the cross section for incident particle
        - particle: uses the energy and particle type (number of electrons) to
          determine which cross section to use, then applies a 1/beta^2 scaling
          of the cross section measured by saha
        """
        # saha et al, Measurement of 181 MeV H- ions stripping cross-sections by
        # carbon stripper foil, NIM A Vol 776 2915
        if a_particle.pid == 9900010010020:
            ref_sigma = 1.580e-18
        elif a_particle.pid == 9900010010010:
            ref_sigma = 0.648e-18
        ref_beta = particle.Particle.new_from_ke(181, 2212).beta_rel
        sigma = ref_sigma*ref_beta**2/a_particle.beta_rel**2
        return sigma

    def stripping_cross_section(self, a_particle):
        """
        Calculate the cross section for incident particle; use the algorithm as
        determined by self.stripping_algorithm
        """
        if self.stripping_algorithm == "saha":
            sigma = self.stripping_cross_section_saha(a_particle)
        elif self.stripping_algorithm == "nakai":
            sigma = self.stripping_cross_section_nakai(a_particle)
        else:
            raise ValueError("Did not recognise stripping algorithm "+\
                             str(self.stripping_algorithm))
        return sigma

    def strip(self, a_particle, step_size):
        """
        Calculate the probability of stripping a particle in a given step
        - a_particle: the particle to strip
        - step_size: the distance traversed through the material
        Uses p = sigma*number_density*step_size (assumes step size is small so
        that events where one then another electron are removed are rare)
        """
        sigma = self.stripping_cross_section(a_particle)
        number_density = self.density/self.molecular_mass*constants.avogadro_constant
        exponent = sigma*number_density*step_size
        #print "SIGMA", sigma, self.molecular_mass
        # probability of surviving step size is exp(-sigma*number_density*step_size)
        probability = 1-math.exp(-exponent)
        return probability

    def double_strip(self, a_particle, step_size):
        """
        Calculate the probability of ionizing a H- in a given step
        - a_particle: the particle to strip
        - step_size: the distance traversed through the material
        Uses p = sigma*number_density*step_size (assumes step size is small so
        that events where one then another electron are removed are rare)
        
        Returns tuple of
        - probability that the particle is unchanged
        - probability that the particle is singly ionized
        - probability that the particle is doubly ionized
        """
        a_particle = copy.deepcopy(a_particle)
        if a_particle.pid != 9900010010020:
            particle_type = constants.get_name(a_particle.pid)
            raise RuntimeError("Cant double strip "+particle_type)
        sigma_m10 = self.stripping_cross_section(a_particle)
        a_particle.pid = 9900010010010
        sigma_01 = self.stripping_cross_section(a_particle)
        number_density = self.density/self.molecular_mass*constants.avogadro_constant
        number = number_density*step_size
        
        p_negative = math.exp(-sigma_m10*number)
        p_neutral = sigma_m10/(sigma_01-sigma_m10)*\
                    (math.exp(-sigma_m10*number) - math.exp(-sigma_01*number))
        p_positive = 1 - p_neutral - p_negative
        return p_negative, p_neutral, p_positive

    def scattering(self, a_particle, thickness):
        """
        PDG model for the scattering angle
        - a_particle: particle to be scattered
        - thickness: thickness (float) of materials; should be a real distance (not normalised to density)
        returns the projected RMS according to the PDG formula
        """
        a_particle = copy.deepcopy(a_particle)
        t0 = 13.6/a_particle.beta_rel/a_particle.get_momentum()
        t0 *= abs(a_particle.charge)*(thickness/self.X0)**0.5
        return t0

    @classmethod
    def set_materials_file_name(cls, file_name):
        cls.materials_file = file_name

    m_e = constants.get_mass(11)
    materials_file = os.path.split(__file__)[0]+"/../../share/foil/materials.json"
    #materials_file = "/home/cr67/OPAL/external_rogers_mint/install/lib/optimisation_tools/share/foil/materials.json"
    e_r = constants.rydberg_energy*constants.get_mass(9900010010010)/constants.get_mass(11)

