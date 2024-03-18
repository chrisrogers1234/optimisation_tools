import matplotlib
import matplotlib.pyplot
import math

class FieldModel(object):
    def get_field_profile(self, phi_list):
        bz_list = [self.get_field(phi) for phi in phi_list]
        return bz_list

class TanhSumField(FieldModel):
    def __init__(self, b0, c1, r0, lambda0, phi0, phi_offset):
        self.phi0 = math.radians(phi0)
        self.b0 = b0
        self.phi_offset = math.radians(phi_offset)
        self.const = c1*r0/lambda0
        print("Built tanh magnet with field const: {0}".format(self.const))

    def get_field(self, phi):
        phi = math.radians(phi)-self.phi_offset
        tanhpos = math.tanh((phi+self.phi0/2.0)*self.const/2.0)
        tanhneg = math.tanh((phi-self.phi0/2.0)*self.const/2.0)
        tanh = self.b0*0.5*(tanhpos-tanhneg) # math.tanh((self.phi0/2.0-phi)*self.c1/self.lambda0))
        return tanh

class EngeProductField(FieldModel):
    def __init__(self, b0, c1, r0, lambda0, phi0, phi_offset):
        self.phi0 = math.radians(phi0)
        self.b0 = b0
        self.phi_offset = math.radians(phi_offset)
        self.const = c1*r0/lambda0
        print("Built enge magnet with field const: {0}".format(self.const))

    def get_field(self, phi):
        phi = math.radians(phi)-self.phi_offset
        phi_entrance = -self.phi0/2.0 - phi
        phi_exit = phi - self.phi0/2.0
        f_entrance = 1.0/(1+math.exp(self.const*phi_entrance))
        f_exit = 1.0/(1+math.exp(self.const*phi_exit))
        f = self.b0*f_entrance*f_exit
        return f

def plot_magnets():
    """"""
    n_points = 100
    c1 = 2.95
    r0 = 4.0
    l0 = 0.07
    phi0_m1 = 2.4
    phi0_m2 = 4.8
    phi_offset_m1 = -1.2-phi0_m1/2.0 # offset by half the magnet opening angle plus half the nominal drift
    phi_offset_m2 = +1.2+phi0_m2/2.0 # offset by half the magnet opening angle plus half the nominal drift
    b0_m1 = 0.1239
    b0_m2 = -0.33

    tanh_one = TanhSumField    (b0_m1, c1, r0, l0, phi0_m1, phi_offset_m1)
    enge_one = EngeProductField(b0_m1, c1, r0, l0, phi0_m1, phi_offset_m1)
    tanh_two = TanhSumField    (b0_m2, c1, r0, l0, phi0_m2, phi_offset_m2)
    enge_two = EngeProductField(b0_m2, c1, r0, l0, phi0_m2, phi_offset_m2)
    phi_max = (phi0_m1+phi0_m2)*5

    phi_list = [(i-n_points/2)*phi_max/n_points for i in range(n_points+1)]
    tanh_one_list = tanh_one.get_field_profile(phi_list)
    enge_one_list = enge_one.get_field_profile(phi_list)
    tanh_two_list = tanh_two.get_field_profile(phi_list)
    enge_two_list = enge_two.get_field_profile(phi_list)
    tanh_sum_list = [tanh_one_list[i] + tanh_two_list[i] for i in range(n_points+1)]
    enge_sum_list = [enge_one_list[i] + enge_two_list[i] for i in range(n_points+1)]

    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(phi_list, tanh_one_list, linestyle='dashed', c='b', label="tanh m1")
    axes.plot(phi_list, enge_one_list, linestyle='dashed', c='r', label="enge m1")
    axes.plot(phi_list, tanh_two_list, linestyle='dotted', c='b', label="tanh m2")
    axes.plot(phi_list, enge_two_list, linestyle='dotted', c='r', label="enge m2")
    axes.plot(phi_list, enge_sum_list, c='b', label="tanh sum")
    axes.plot(phi_list, enge_sum_list, c='r', label="enge sum")
    axes.set_xlabel("$\\phi$ [$^\\circ$]")
    axes.set_ylabel("B [T]")
    axes.legend()
    figure.savefig("analytical_field_profile.png")

def main():
    plot_magnets()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")