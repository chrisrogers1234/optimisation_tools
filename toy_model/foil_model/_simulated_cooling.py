import math
import ROOT
import xboa.common as common

root_objects = []

class CoolingChannel:
    def __init__(self, material, momentum):
        self.cell_length = 860.
        self.mass = 105.658
        self.charge = 1.
        self.rho = 0.69
        self.thickness = 35. # cm
        self.beta_hill = 420. # mm
        self.set_p(momentum)
        self.set_material(material)

        #self.dEdz =  1.940*self.rho # MeV/cm #1.850 #

    def set_material(self, material_name):
        if material_name == "beryllium":
            self.X0 =  35.28*common.units['cm'] # cm
            self.density =  1.848  # g/cm^3
            self.ion_energy =  63.7*common.units['eV'] # eV
            self.elements_list = [{"name":"beryllium", "z":4, "a":9.0121831, "fraction":1.}]
        elif material_name == "liquid_hydrogen":
            self.X0 = 890.4*common.units['cm']
            self.density = 0.07080
            self.ion_energy = 21.8*common.units['eV']
            self.elements_list = [{"name":"hydrogen", "z":1, "a":1.00794, "fraction":1}]
        elif material_name == "gaseous_hydrogen":
            self.X0 = 890.4*common.units['cm']
            self.density = 8.376E-05
            self.ion_energy = 21.8*common.units['eV']
            self.elements_list = [{"name":"hydrogen", "z":1, "a":1.00794, "fraction":1}]
        elif material_name == "carbon":
            self.X0 = 19.32*common.units['cm']
            self.density = 2.210
            self.ion_energy = 78.0*common.units['eV']
            self.elements_list = [{"name":"carbon", "z":6, "a":12.0107, "fraction":1}]
        elif material_name == "aluminium":
            self.X0 =  8.897*common.units['cm']
            self.density =  2.699
            self.ion_energy =  166.0*common.units['eV']
            self.elements_list = [{"name":"aluminium", "z":13, "a":26.9815385, "fraction":1}]
        elif material_name == "lithium_hydride":
            self.X0 = 97.09*common.units['cm'] # cm
            self.density = 0.8200  # g/cm^3
            self.ion_energy = 36.5*common.units['eV'] # eV
            self.elements_list = [
                {"name":"lithium", "z":3, "a":6.94, "fraction": 0.873203},
                {"name":"hydrogen", "z":1, "a":1.00794, "fraction": 0.126797},
            ]

    def dEdz(self):
        bg2 = self.beta_rel**2*self.gamma_rel**2
        m_ratio = self.m_e/self.mass
        t_max = 2*self.m_e*bg2/(1+2*self.gamma_rel*m_ratio+m_ratio**2)

        coefficient = math.log(2*self.m_e*bg2*t_max/(self.ion_energy**2)) - self.beta_rel**2
        coefficient *= 0.307075*self.charge**2/2/self.beta_rel**2*self.density/common.units['cm']

        dEdz = 0.
        for element in self.elements_list:
            dEdz += coefficient * element["z"]/element["a"]*element["fraction"]
        return -dEdz

    def set_p(self, new_p):
        self.p = new_p
        self.E = (self.p**2+self.mass**2)**0.5
        self.gamma_rel = self.E/self.mass
        self.beta_rel = self.p/self.mass/self.gamma_rel

    def set_ke(self, new_ke):
        self.E = new_ke + self.mass
        self.gamma_rel = self.E/self.mass
        self.p = (self.E**2-self.mass**2)**0.5
        self.beta_rel = self.p/self.mass/self.gamma_rel

    def depsdz(self, eps_z0):
        energy_loss = self.dEdz()
        deps = eps_z0/self.beta_rel**2/self.E*self.dEdz()*self.thickness # cm
        dscatter = self.beta_hill*13.6**2/2/self.beta_rel**3/self.E/self.mass/self.X0*self.thickness # cm
        return (deps+dscatter)

    def eqm_emittance(self):
        eqm_emit = common.nd_newton_raphson1(
                  lambda x: [self.depsdz(x[0])],
                  [1e-3],
                  [1.],
                  [0.1],
                  verbose = False)[0]
        return eqm_emit

    def n_less_than_amp(self, amplitude, emittance):
        return ROOT.Math.chisquared_cdf(amplitude/emittance, 4)

    m_e = common.pdg_pid_to_mass[11]

def test_energy_loss():
    material = "beryllium"
    for p, dEdz in [(1.527E+02, 1.940), (1.994E+02, 1.743E+00), (2.218E+02, 1.694E+00)]: # MeV/g cm ^-2
        cooling_channel = CoolingChannel(material, p)
        print(material, p, dEdz, cooling_channel.dEdz(), cooling_channel.dEdz()/cooling_channel.density)
    material = "liquid_hydrogen"
    for p, dEdz in [(1.527E+02, 1.940), (1.994E+02, 1.743E+00), (2.218E+02, 1.694E+00)]: # MeV/g cm ^-2
        cooling_channel = CoolingChannel(material, p)
        print(material, p, dEdz, cooling_channel.dEdz(), cooling_channel.dEdz()/cooling_channel.density)

def dbeta_graph(material, eps, momentum, thickness, canvas = None, line_color=1):
    cooler = CoolingChannel(material, momentum)
    cooler.mass = common.pdg_pid_to_mass[13]
    beta_list = list(range(20, 2000))
    deps_list = []
    for beta in beta_list:
        cooler.beta_hill = beta
        deps_list.append(cooler.depsdz(eps)/eps*thickness)
    hist, graph = common.make_root_graph('#epsilon_{n} = '+str(eps)+' mm', beta_list, '#beta_{#perp}    [mm]', deps_list, 'Fractional change in emittance [%]')
    if canvas == None:
        canvas = common.make_root_canvas('emittance change')
        canvas.Draw()
        hist.Draw()
    canvas.cd()
    graph.SetLineColor(line_color)
    graph.Draw("same")
    canvas.Update()
    return canvas, hist, graph

def demit_graph(material, mom, thickness):
    emittance_canvas = None
    graph_list = []
    for eps, color in [(3., ROOT.kRed), (6., ROOT.kRed+2), (10., 1)]:
        emittance_canvas, hist, graph = dbeta_graph(material, eps, mom, thickness, emittance_canvas, color)
        hist.SetTitle(str(thickness)+" mm "+material+" p = "+str(mom))
        graph_list.append(graph)
    common.make_root_legend(emittance_canvas, graph_list)
    emittance_canvas.Update()
    emittance_canvas.Print("demit_"+material+"_"+str(mom)+".eps")


def eqm_graph(material):
      channel = CoolingChannel(material, 11.)
      channel.mass = common.pdg_pid_to_mass[2212]
      canvas = None
      ke_list = [1]+list(range(5, 31, 5))
      graph_list = []
      print(material)
      for i, ke in enumerate(ke_list):
          i_rat = i/float(len(ke_list))
          if i_rat > 0.5:
              color = ROOT.TColor.GetColor(0., i_rat, (1.-i_rat)*2)
          else:
              color = ROOT.TColor.GetColor(i_rat*2, 0., 0.)
          channel.set_ke(ke)
          print(ke, channel.p, channel.dEdz())
          beta_list = [float(i) for i in range(100, 5001, 100)]
          eqm_list = []
          for beta in beta_list:
              channel.beta_hill = beta
              eqm_list.append(channel.eqm_emittance())
          #print beta_list, eqm_list
          hist, graph = common.make_root_graph("KE = "+str(ke)+" MeV",
                                               beta_list, "#beta [mm]",
                                               eqm_list, "#varepsilon_{n} [mm]",
                                               ymin = 0., ymax = 3.)
          if canvas == None:
              canvas = common.make_root_canvas("eqm_graph - "+material)
              canvas.Draw()
              hist.SetTitle(material)
              hist.Draw()
          graph.SetLineColor(color)
          graph.Draw("SAME")
          graph_list.append(graph)
      common.make_root_legend(canvas, graph_list)
      canvas.Update()

def chi_squared_cdf():
    for amplitude in range(0, 73):
        print(amplitude, CoolingChannel(MATERIAL).n_less_than_amp(float(amplitude), 6.))

MATERIAL = "beryllium"



def main():
    #test_energy_loss()
    eqm_graph("beryllium")
    eqm_graph("carbon")
    eqm_graph("liquid_hydrogen")
    eqm_graph("gaseous_hydrogen")

    demit_graph("liquid_hydrogen", 140., 350.)
    demit_graph("aluminium", 140., 1.)
    demit_graph("lithium_hydride", 140., 65.)

    #emittance_canvas.Print("plots/emittance_vs_demittance.png")
    #emittance_canvas = dp_graph()
    #emittance_canvas.Print("plots/emittance_vs_dp.png")
    return
    emittance_canvas, hist, graph_3 = dbeta_graph(3, None, 2)
    emittance_canvas, hist, graph_6 = dbeta_graph(6, emittance_canvas, 4)
    emittance_canvas, hist, graph_12= dbeta_graph(12, emittance_canvas, 6)
    common.make_root_legend(emittance_canvas, [graph_3, graph_6, graph_12])
    emittance_canvas.Print("plots/emittance_vs_dbeta.png")
    emittance_canvas.Print("plots/emittance_vs_dbeta.root")

if __name__ == "__main__":
    main()
    input()
