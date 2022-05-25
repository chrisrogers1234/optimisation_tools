import os
import shutil
from toy_model.foil_model.material import Material
from toy_model.foil_model.particle import Particle
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot
import numpy.random

def setup_foil():
    material = Material()
    material.set_material("carbon")
    return material

def energy_loss(material):
    def convert_axes_dedzrho(axes_dedz): # "closure function"
        rho = material.density
        y1, y2 = axes_dedz.get_ylim()
        axes_dedzrho.set_ylim(y1/rho, y2/rho)
        axes_dedzrho.figure.canvas.draw()

    p_list = [p for p in range(50, 10001)]
    dedz_list = []
    for p in p_list:
        particle = Particle.new_from_momentum(p, 2212)
        dedz = abs(material.energy_loss_dz(particle))
        dedz_list.append(dedz)
    fig = matplotlib.pyplot.figure(figsize=(20, 10))
    axes_dedzrho = fig.add_subplot(1, 1, 1, xscale="log", yscale="log", position=[0.1, 0.1, 0.0001, 0.8]) #matplotlib.axes.Axes(fig, [0.1, 0.1, 0.8, 0.8], sharex=axes_dedz)
    axes_dedzrho.set_ylabel("dE/dx [MeV g$^{-1}$ cm$^{2}$", fontsize=20)
    axes_dedzrho.tick_params(axis='x', which='both', labelcolor='w')
    axes_dedzrho.tick_params(axis='y', labelsize = 14)

    axes_dedz = fig.add_subplot(1, 1, 1, xscale="log", yscale="log", position=[0.2, 0.1, 0.7, 0.8])
    axes_dedz.set_xlabel("Momentum [MeV/c]", fontsize=20)
    axes_dedz.set_ylabel("dE/dx [MeV cm$^{-1}$", fontsize=20)
    axes_dedz.tick_params(axis='both', labelsize = 14)

    axes_dedz.set_xticks([100., 1000., 10000.], True)
    axes_dedz.set_xticks([i*10 for i in range(5, 9)]+
                         [i*100 for i in range(1, 9)]+
                         [i*1000 for i in range(1, 9)]+
                         [i*10000 for i in range(1, 10)], False)
    axes_dedz.callbacks.connect("ylim_changed", convert_axes_dedzrho)
    axes_dedz.plot(p_list, dedz_list)
    return fig

def momentum_change(material, kinetic_energy, column_density):
    particle_0 = Particle.new_from_ke(kinetic_energy, 2212)
    dedz = material.energy_loss_dz(particle_0)
    de_1 = dedz*column_density/material.density
    particle_1 = Particle.new_from_ke(kinetic_energy+de_1, 2212)
    dp_over_p_1 = (particle_1.p-particle_0.p)/particle_0.p
    print("For material with column density", column_density, "and incident KE", kinetic_energy, "dE/E", de_1/kinetic_energy, "dp/p", dp_over_p_1)
    return dp_over_p_1

def energy_straggling(material, kinetic_energy, column_density):
    #particle_0 = Particle.new_from_ke(kinetic_energy, 2212)
    material = Material()
    material.set_material("liquid_hydrogen")
    thickness = 35.0 # cm
    column_density = 35.0 *material.density

    particle_0 = Particle.new_from_momentum(200.0, 13)
    dE = material.energy_loss_dz(particle_0)*thickness
    mu = material.energy_straggling_moments(particle_0, column_density)
    print("dE:", dE, "mu:", mu)
    for i in range(51):
        delta = (i-5.0)/5.0
        p = material.energy_straggling(particle_0, column_density, delta)
        print(format(delta, "6.4g"), p)
    print(material.straggling_hermite)
    #f = material.energy_straggling_distribution(particle_0, column_density)
    #for delta in range(100):
    #    dE = delta/100.0
    #    print(format(dE, "8.4g"), format(f(dE), "8.4g"))

def scattering(material):
    particle = Particle.new_from_momentum(75.0, 2212)
    thickness_1 = 20e-6/material.density
    thickness_2 = 5e-6/material.density
    sigma_1 = material.scattering(particle, thickness_1)
    sigma_2 = material.scattering(particle, thickness_2)
    fig = matplotlib.pyplot.figure(figsize=(20, 10))
    axes = fig.add_subplot(1, 1, 1)
    scatters = numpy.random.randn(100000000)*sigma_1*1e3
    axes.hist(scatters, bins=500, histtype='step', density=True, label = "$20 \\times 10^{-6}$ g cm$^{-2}$ with standard deviation "+format(sigma_1*1e3, "6.3g")+" mrad")
    scatters = numpy.random.randn(100000000)*sigma_2*1e3
    axes.hist(scatters, bins=500, histtype='step', density=True, label = "$5 \\times 10^{-6}$ g cm$^{-2}$ with standard deviation "+format(sigma_2*1e3, "6.3g")+" mrad")
    axes.set_xlabel("Angle [mrad]", fontsize=20)
    axes.set_ylabel("Number density [mrad$^{-1}$]", fontsize=20)
    axes.tick_params(axis='both', labelsize = 14)
    axes.legend(fontsize=20)

    return fig

def foil_test():
    out_dir = "output/toy_model_tests/foil/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    foil = setup_foil()
    #fig = energy_loss(foil)
    #fig.savefig(out_dir+"foil_test_dedx.png")
    #fig = scattering(foil)
    #fig.savefig(out_dir+"foil_test_scattering.png")
    #dp_over_p_1 = momentum_change(foil, 3.0, 5e-6)
    #dp_over_p_2 = momentum_change(foil, 3.0, 20e-6)
    #print("Ratio", dp_over_p_1/dp_over_p_2)
    energy_straggling(foil, 1000.0, 20e-6)

if __name__ == "__main__":
    foil_test()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
