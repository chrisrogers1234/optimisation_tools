import copy
import sys
import math
import os
import numpy
import scipy
from scipy.integrate import odeint, solve_ivp
import matplotlib
import ROOT

import xboa.common
import xboa.hit

import PyOpal.parser
import PyOpal.field

class PyOpalTracking(object):
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        self.setup()

    def setup(self):
        self.charge = 1.0
        self.c_light = xboa.common.constants["c_light"]
        self.u_angle = 0.0
        self.figure = matplotlib.pyplot.figure(figsize=(20, 10))
        self.axes0 = self.figure.add_subplot(1, 2, 1)
        self.axes1 = self.figure.add_subplot(1, 2, 2)
        self.load_lattice(self.config.tracking["lattice_file_out"])
        self.plot_field_2d(self.axes0, 100, [0.0, 5000.0], [0.0, 5000.0])
        self.atol = self.config.tracking["py_tracking"]["atol"]
        self.rtol = self.config.tracking["py_tracking"]["rtol"]
        self.verbose = self.config.tracking["py_tracking"]["verbose"]
        der = self.config.tracking["py_tracking"]["derivative_function"]
        if der == "u_derivative":
            self.derivative_function = self.u_derivative
        elif der == "t_derivative":
            self.derivative_function = self.t_derivative
        elif der == "y_derivative":
            self.derivative_function = self.y_derivative
        else:
            raise ValueError("Could not parse derivative_function "+str(der))
        self.step_list = []

        self.step_number = 0
        self.y_out = []
        self.y_tracking = []

    def load_lattice(self, lattice_file):
        if (self.initialised):
            raise RuntimeError("Attempt to reinitialise opal")
        self.initialised = True
        here = os.getcwd()
        os.chdir(self.run_dir)
        fout = open("disttest.dat", "w")
        fout.write("1\n0.0 0.0 0.0 0.0 0.0 0.0")
        fout.close()
        PyOpal.parser.initialise_from_opal_file(lattice_file)
        os.chdir(here)

    def track_single_step(self):
        c = 1.
        y0 = self.y_out[-1]
        if self.derivative_function == self.t_derivative:
            beta = (y0[5]**2+y0[6]**2+y0[7]**2)**0.5/y0[4]
            c = beta/self.c_light
        if self.step_number == 0:
            t_list = [0.0, self.step_list[self.step_number]] 
        else:
            t_list = [self.step_list[self.step_number-1], self.step_list[self.step_number]] 
        if self.derivative_function == self.u_derivative:
            self.u_angle = t_list[-1]
            y0_rot = self.rotate_8(y0, self.u_angle)
            if y0_rot[2] > 0.0:
                print("y0:", y0, "rotated y0", y0_rot)
                raise RuntimeError("Attempt to track through more than 180 degrees (or less than 0)")
            y0 = y0_rot
            t_list = [0.0, abs(y0[2])]
        if self.verbose:
            print("Track single step", self.u_angle, t_list, y0, self.rtol*c, self.atol*c)
        if abs(t_list[1] - t_list[0]) < c:
            this_y_out = solve_ivp(self.derivative_function, t_list, y0, rtol=self.rtol*c, atol=self.atol*c)["y"]
        else:            
            this_y_out = solve_ivp(self.derivative_function, t_list, y0, rtol=self.rtol*c, atol=self.atol*c, first_step=c)["y"]
        this_y_out = numpy.transpose(this_y_out)
        if self.derivative_function == self.u_derivative:
            self.y_out += [self.rotate_8(row, -self.u_angle) for row in this_y_out]
        else:
            self.y_out += this_y_out.tolist()
        self.y_tracking += [self.y_out[-1]]

    def convert_to_cylindrical(self, hit):
        phi = math.atan2(hit['z'], hit['x'])
        x, y = hit["x"], hit["z"]
        px, py = hit["px"], hit["pz"]
        hit["x"] = + x*math.cos(phi) + y*math.sin(phi)
        hit["z"] = - x*math.sin(phi) + y*math.cos(phi)
        hit["px"] = + px*math.cos(phi) + py*math.sin(phi)
        hit["pz"] = + px*math.sin(phi) - py*math.cos(phi)

    def track_one(self, hit):
        self.y_out = [[hit[var] for var in self.var_list]]
        self.y_tracking = []
        if self.verbose:
            print("Track one", self.step_list, self.y_out)

        for self.step_number in range(len(self.step_list)):
            self.track_single_step()
        hit_list = []
        for output in self.y_tracking:
            my_hit = hit.deepcopy()
            for i, var in enumerate(self.var_list):
                my_hit[var] = output[i]
            self.convert_to_cylindrical(my_hit)
            hit_list.append(my_hit)
        return hit_list

    def track_many(self, hit_list):
        hit_list_of_hits = [self.track_one(hit) for hit in hit_list]
        return hit_list_of_hits
    
    @classmethod
    def rotate_8(cls, y, angle):
        y_rotated = copy.deepcopy(y)
        rot_xy   = cls.rotate_2([y[1], y[2]], angle)
        rot_pxpy = cls.rotate_2([y[5], y[6]], angle)
        y_rotated = [y[0],
                     rot_xy[0], 
                     rot_xy[1],
                     y[3],
                     y[4],
                     rot_pxpy[0], 
                     rot_pxpy[1], 
                      y[7]
              ]
        return y_rotated

    @classmethod
    def rotate_2(cls, y, angle):
        y_rotated = [+y[0]*math.cos(angle)+y[1]*math.sin(angle), 
                     -y[0]*math.sin(angle)+y[1]*math.cos(angle)]
        return y_rotated

    @classmethod
    def get_field(cls, x):
        field = PyOpal.field.get_field_value(x[1]/1000., x[2]/1000., x[3]/1000., x[0])
        field = (field[1]*1e-3, field[2]*1e-3, field[3]*1e-3, field[4], field[5], field[6])
        return field

    def u_derivative(self, z, x_local):
        x_global = self.rotate_8(x_local, -self.u_angle)
        field = self.get_field(x_global)
        rot_bxby   = self.rotate_2([field[0], field[1]], self.u_angle)
        rot_exey   = self.rotate_2([field[3], field[4]], self.u_angle)
        field = (rot_bxby[0], 
                 rot_bxby[1],
                 field[2],
                 rot_exey[0],
                 rot_exey[1],
                 field[5])

        energy = x_local[4]
        dxdz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # make sure energy cannot go below mass
        # dx/dt = px/E
        dxdz[0] = x_local[4]/x_local[6]/self.c_light
        dxdz[1] = x_local[5]/x_local[6]
        dxdz[2] = x_local[6]/x_local[6]
        dxdz[3] = x_local[7]/x_local[6]

        # dp/dt = qE + vxB
        dxdz[5] = self.charge*self.c_light*( dxdz[2]*field[2] - dxdz[3]*field[1] ) + self.charge*self.c_light*field[3]*dxdz[0]
        dxdz[6] = self.charge*self.c_light*( dxdz[3]*field[0] - dxdz[1]*field[2] ) + self.charge*self.c_light*field[4]*dxdz[0]
        dxdz[7] = self.charge*self.c_light*( dxdz[1]*field[1] - dxdz[2]*field[0] ) + self.charge*self.c_light*field[5]*dxdz[0]

        # E^2  = px^2+py^2+pz^2+ m^2
        # E dE = px dpx + py dpy + pz dpz
        # dE/dt = dx/dz dpx + dy/dz dpy + dz/dz dpz (ignore B as B conserves energy)
        dxdz[4] = self.charge*field[3]*dxdz[1]+self.charge*field[4]*dxdz[2]+self.charge*field[5]*dxdz[3]
        if self.verbose and False:
            print(format(z, "10.6g"), end=" ")
            for i, value in enumerate(list(x_local)+dxdz+list(field)[:3]):
                if (i) % 4 == 0:
                    print(" ** ", end=" ")
                print(format(value, "8.4g"), end=" ")
            print()
        return dxdz

    def y_derivative(self, z, x):
        field = self.get_field(x)

        energy = x[4]
        dxdz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # make sure energy cannot go below mass
        # dx/dt = px/E
        dxdz[0] = x[4]/x[6]/self.c_light
        dxdz[1] = x[5]/x[6]
        dxdz[2] = x[6]/x[6]
        dxdz[3] = x[7]/x[6]

        # dp/dt = qE + vxB
        dxdz[5] = self.charge*self.c_light*( dxdz[2]*field[2] - dxdz[3]*field[1] ) + self.charge*self.c_light*field[3]*dxdz[0]
        dxdz[6] = self.charge*self.c_light*( dxdz[3]*field[0] - dxdz[1]*field[2] ) + self.charge*self.c_light*field[4]*dxdz[0]
        dxdz[7] = self.charge*self.c_light*( dxdz[1]*field[1] - dxdz[2]*field[0] ) + self.charge*self.c_light*field[5]*dxdz[0]

        # E^2  = px^2+py^2+pz^2+ m^2
        # E dE = px dpx + py dpy + pz dpz
        # dE/dt = dx/dz dpx + dy/dz dpy + dz/dz dpz (ignore B as B conserves energy)
        dxdz[4] = self.charge*field[3]*dxdz[1]+self.charge*field[4]*dxdz[2]+self.charge*field[5]*dxdz[3]

        return dxdz

    def t_derivative(self, t, x):
        field = self.get_field(x)

        energy = x[4]
        dxdt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # make sure energy cannot go below mass
        # dx/dt = px/E
        dxdt[1] = self.c_light*x[5]/energy;
        dxdt[2] = self.c_light*x[6]/energy;
        dxdt[3] = self.c_light*x[7]/energy;
        # dt/dt = 1.
        dxdt[0] = 1.;
        # dp/dt = qE + vxB
        dxdt[5] = self.charge*self.c_light*( dxdt[2]*field[2] - dxdt[3]*field[1] ) + self.charge*self.c_light*field[3];
        dxdt[6] = self.charge*self.c_light*( dxdt[3]*field[0] - dxdt[1]*field[2] ) + self.charge*self.c_light*field[4];
        dxdt[7] = self.charge*self.c_light*( dxdt[1]*field[1] - dxdt[2]*field[0] ) + self.charge*self.c_light*field[5];
        # E^2  = px^2+py^2+pz^2+ m^2
        # E dE = px dpx + py dpy + pz dpz
        # dE/dt = dx/dt dpx + dy/dt dpy + dz/dt dpz (ignore B as B conserves energy)
        dxdt[4] = self.charge*field[3]*dxdt[1]+self.charge*field[4]*dxdt[2]+self.charge*field[5]*dxdt[3];

        return dxdt

    def plot_field_2d(self, axes, n_bins, x_range, y_range):
        x_list = []
        y_list = []
        b_list = []
        for i in range(n_bins):
            sys.stdout.flush()
            for j in range(n_bins):
                x = (i+0.5)/(n_bins+1)*(x_range[1]-x_range[0])+x_range[0]
                y = (j+0.5)/(n_bins+1)*(y_range[1]-y_range[0])+y_range[0]
                x_list.append(x)
                y_list.append(y)
                oob, bx, by, bz, ex, ey, ez = PyOpal.field.get_field_value(x/1000.0, y/1000.0, 0.0, 0.0)
                b_list.append(bz)
        axes.hist2d(x_list, y_list, range=(x_range, y_range), bins=100, weights=b_list)

    @classmethod
    def multisort(cls, list_1, list_2):
        zlist = zip(list_1, list_2)
        print(zlist)
        zlist = sorted(zlist)
        list_1, list_2 = zip(*zlist)
        return list_1, list_2

    def plot_tracking(self, name):
        x_list = [row[1] for row in self.y_out]
        y_list = [row[2] for row in self.y_out]
        self.axes0.plot(x_list, y_list, label=name)
        self.axes0.legend()

        phi_list = [math.degrees(math.atan2(row[2], row[1])) for row in self.y_out]
        z_list = [row[3] for row in self.y_out]
        phi_list, z_list = self.multisort(phi_list, z_list)
        self.axes1.plot(phi_list, z_list, label=name)
        self.axes1.legend()

    def print_tracking(self):
        print("Tracking output")
        for header in ["t", "x", "y", "z", "E", "px", "py", "pz"]:
            print(header.rjust(10))
        for row in self.y_out:
            for cell in row:
                print(format(cell, "10.4g"), end = " ")
            print()

    def find_co(self, seed_hit, opt_vars, opt_errs, n_iterations, target_score):
        self.seed_hit = seed_hit
        self.opt_vars = opt_vars
        self.opt_errs = opt_errs
        self.iteration_number = 1
        self.minuit = ROOT.TMinuit(len(opt_vars))
        for i, var in enumerate(opt_vars):
            self.minuit.DefineParameter(i, var, seed_hit[var], 1e-1, 0, 0)
        self.minuit.SetFCN(self.minuit_function)
        self.minuit.Command("SIMPLEX "+str(n_iterations)+" "+str(target_score))
        return self.get_minuit_hit()

    def get_minuit_hit(self):
        hit = self.seed_hit.deepcopy()
        for i, var in enumerate(self.opt_vars):
            x = ROOT.Double()
            err = ROOT.Double()
            self.minuit.GetParameter(i, x, err)
            hit[var]  = float(x)
        return hit


    def minuit_function(self, nvar, parameters, score, jacobian, err):
        hit_in = self.get_minuit_hit()
        hit_out = self.track_one(hit_in)[-1]
        print("Iteration", self.iteration_number)
        for hit in hit_in, hit_out:
            for var in self.var_list:
                print(var.ljust(10), format(hit[var], "14.10g"), end=" ")
            print()
        score[0] = 0.0
        for var in self.opt_vars:
            score[0] += (hit_in[var]-hit_out[var])**2/self.opt_errs[var]**2
        score[0]**0.5
        self.iteration_number += 1

    initialised = False
    var_list = ["t", "x", "z", "y", "energy", "px", "pz", "py"]

class MockConfig(object):
    def __init__(self):
        self.tracking = {
            "lattice_file_out":"VerticalSectorFFA.tmp",
            "py_tracking":{
                "derivative_function":"u_derivative",
                "atol":1e-15,
                "rtol":1e-15,
                "verbose":False,
            }
        }


def main():
    mass = xboa.common.pdg_pid_to_mass[2212]
    E = mass+3.0
    y0 = {"t":0.0, "x":3739.4181596237268, "y":-88.82695590394741, "z":0.0, "mass":mass, "px":0.0004091193948864643, "pz":75.09082484452648, "py":-3.6067815651168215e-05, "energy":E}
    y0 = {"t":0.0, "x":3739.4181596237268, "y":-88.82695590394741, "z":0.0, "mass":mass, "px":0.0, "pz":75.09082484452648, "py":0.0, "energy":E}
    hit0 = xboa.hit.Hit.new_from_dict(y0, "energy")
    t_list = numpy.linspace(0, 101.6, 2)
    y_list = numpy.linspace(0, 2180.0, 2)
    config = MockConfig()
    tracking = PyOpalTracking(config, "output/double_triplet_baseline/baseline/tmp/find_closed_orbits/")
    """
    print("Track t")
    tracking.step_list = [101.6/2*i for i in range(3)]
    tracking.derivative_function = tracking.t_derivative
    tracking.track_one(hit0)
    tracking.plot_tracking("t")
    print("Track y")
    tracking.step_list = [2180.0/2*i for i in range(3)]
    tracking.derivative_function = tracking.y_derivative
    tracking.track_one(hit0)
    tracking.plot_tracking("y")
    """
    print("Track u")
    tracking.step_list = [i/10.0*2.0*math.pi for i in range(2)]
    tracking.derivative_function = tracking.u_derivative
    hit_list = tracking.track_one(hit0)
    for hit in hit_list:
        for var in tracking.var_list:
            print(var.ljust(10), hit[var])
    list_1 = [0.0, -1.0, 1.0]
    list_2 = [5.0, 0.0, 10.0]
    #print(list_1, list_2, tracking.multisort(list_1, list_2))
    co_hit = tracking.find_co(hit0, ["x", "y"], {"x":1.0, "y":1.0, "x'":0.01, "y'":0.01}, 100, 1e-20) #, "x'", "y'"
    tracking.step_list = [i/10.0*2.0*math.pi for i in range(11)]
    tracking.track_one(co_hit)
    tracking.plot_tracking("y")

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block = False)
    #input("Press <CR> to finish")


