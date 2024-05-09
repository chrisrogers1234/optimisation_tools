import os
import shutil
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot
import numpy.random
from optimisation_tools.toy_model.foil_model.material import Material
from optimisation_tools.toy_model.foil_model.particle import Particle

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
    print("dE/dx @ 3 MeV", material.energy_loss_dz(Particle.new_from_ke(3, 2212)))
    print("dE/dx @ 400 MeV", material.energy_loss_dz(Particle.new_from_ke(400, 2212)))
    print("dE @ 3 MeV", material.energy_loss_dz(Particle.new_from_ke(3, 2212))*20e-6*material.density/3)
    print("dE @ 400 MeV", material.energy_loss_dz(Particle.new_from_ke(400, 2212))*0.0016*material.density/400)
    fig = matplotlib.pyplot.figure()
    axes_dedzrho = fig.add_subplot(1, 1, 1, xscale="log", yscale="log", position=[0.1, 0.1, 0.0001, 0.8]) #matplotlib.axes.Axes(fig, [0.1, 0.1, 0.8, 0.8], sharex=axes_dedz)
    axes_dedzrho.set_ylabel("dE/dx [MeV g$^{-1}$ cm$^{2}$]", fontsize=12)
    axes_dedzrho.tick_params(axis='x', which='both', labelcolor='w')
    axes_dedzrho.tick_params(axis='y', labelsize = 8)

    axes_dedz = fig.add_subplot(1, 1, 1, xscale="log", yscale="log", position=[0.2, 0.1, 0.7, 0.8])
    axes_dedz.set_xlabel("Momentum [MeV/c]", fontsize=12)
    axes_dedz.set_ylabel("dE/dx [MeV cm$^{-1}$]", fontsize=12)
    axes_dedz.tick_params(axis='both', labelsize = 8)

    #axes_dedz.set_xticks([100., 1000., 10000.], True)
    #axes_dedz.set_xticks([i*10 for i in range(5, 9)]+
    #                     [i*100 for i in range(1, 9)]+
    #                     [i*1000 for i in range(1, 9)]+
    #                     [i*10000 for i in range(1, 10)], False)
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
    particle_1 = Particle.new_from_momentum(75.0, 2212)
    particle_2 = Particle.new_from_ke(500.0, 2212)
    thickness_1 = 20e-6/material.density
    thickness_2 = 0.0016/material.density
    sigma_1 = material.scattering(particle_1, thickness_1)
    sigma_2 = material.scattering(particle_2, thickness_2)
    fig = matplotlib.pyplot.figure()
    axes = fig.add_subplot(1, 1, 1)
    scatters = numpy.random.randn(100000000)*sigma_1*1e3
    axes.hist(scatters, bins=500, histtype='step', density=True, label = "3 MeV, 20e-6 g cm$^{-2}$"+format(sigma_1*1e3, "6.3g")+" mrad") #  "+format(sigma_1*1e3, "6.3g")+" mrad"
    scatters = numpy.random.randn(100000000)*sigma_2*1e3
    axes.hist(scatters, bins=500, histtype='step', density=True, label = "400 MeV, 0.0016 g cm$^{-2}$"+format(sigma_2*1e3, "6.3g")+" mrad")
    axes.set_xlabel("Angle [mrad]", fontsize=15)
    axes.set_ylabel("Number density [mrad$^{-1}$]", fontsize=15)
    axes.tick_params(axis='both', labelsize = 12)
    axes.legend(fontsize=12)

    return fig

def stripping(material, kinetic_energy, max_column_density):
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    a_particle = Particle.new_from_ke(kinetic_energy, 9900010010020)
    col_density_list = []
    p0_list = []
    p1_list = []
    p2_list = []
    for column_density in [i*max_column_density/50 for i in range(0, 51)]:
        thickness = column_density/material.density
        p0, p1, p2 = material.double_strip(a_particle, thickness)
        col_density_list.append(column_density)
        p0_list.append(p0)
        p1_list.append(p1)
        p2_list.append(p2)
    axes.plot(col_density_list, p0_list, label="H$^-$")
    axes.plot(col_density_list, p1_list, label="H$^0$")
    axes.plot(col_density_list, p2_list, label="H$^+$")
    axes.legend()
    axes.set_xlabel("Column density [g/cm$^2$]")
    axes.set_ylabel("Probability")
    axes.text(0.4, 0.8, f"Unstripped fraction ({kinetic_energy} MeV):", transform=axes.transAxes)
    col_dens_1 = 5e-6*max_column_density/50e-7
    p0, p1, p2 = material.double_strip(a_particle, col_dens_1/material.density)
    axes.text(0.4, 0.75, f"  @ {col_dens_1} g/cm$^2$: {1-p2:8.4g}", transform=axes.transAxes)
    col_dens_2 = 20e-6*max_column_density/50e-7
    p0, p1, p2 = material.double_strip(a_particle, col_dens_2/material.density)
    axes.text(0.4, 0.70, f"  @ {col_dens_2} g/cm$^2$: {1-p2:8.4g}", transform=axes.transAxes)
    axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useOffset=False)
    return figure



def foil_test():
    out_dir = "output/toy_model_tests/foil/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    foil = setup_foil()
    fig = energy_loss(foil)
    fig.savefig(out_dir+"foil_test_dedx.png")
    fig = scattering(foil)
    fig.savefig(out_dir+"foil_test_scattering.png")
    #dp_over_p_1 = momentum_change(foil, 3.0, 5e-6)
    #dp_over_p_2 = momentum_change(foil, 3.0, 20e-6)
    #print("Ratio", dp_over_p_1/dp_over_p_2)
    #energy_straggling(foil, 1000.0, 20e-6)
    fig = stripping(foil, 400.0, 4000e-7)
    fig.savefig(out_dir+"foil_test_stripping_400_MeV.png")
    fig = stripping(foil, 3.0, 50e-7)
    fig.savefig(out_dir+"foil_test_stripping_3_MeV.png")

if __name__ == "__main__":
    foil_test()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
