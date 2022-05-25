import threading
import os
import sys
import math
import scipy.integrate

import numpy
import matplotlib
import matplotlib.pyplot

import PyOpal.parser
import PyOpal.field


class Integrator3D(object):
    def __init__(self, field_map, start_vector, tsteps):
        self.field_map = field_map
        self.start_vector = start_vector
        self.tsteps = tsteps
        self.point_list = []
        if len(self.tsteps) < 2:
            raise RuntimeError("Tsteps must be at least length 2")

    def get_field_lines(self):
        """
        For some initial set of vectors; integrate along field lines to get
        a current surface.
        """
        self.point_list = []
        print("Calculating points")
        for vec in self.start_vector:
            y = numpy.ndarray(shape=(0, 3))
            for tstep in self.tsteps:
                try:
                    y += scipy.integrate.odeint(get_field_line_derivative,
                                               vec,
                                               [tstep],
                                               (self.field_map,),
                                               full_output=0)
                    self.point_list.append(y)
                    #print(y)
                except Exception:
                    sys.excepthook(*sys.exc_info())
        #print(self.point_list[0][0], self.point_list[0][0])
        #print(self.point_list[-1][0], self.point_list[-1][-1])

    def point_pred(self, point, axis_range):
        in_graph = True
        for i in range(3):
            in_graph = in_graph and (axis_range[2*i] is None or point[i] > axis_range[2*i])
            in_graph = in_graph and (axis_range[2*i+1] is None or point[i] < axis_range[2*i+1])
        return in_graph

    def plot_field_line_3d(self, axis_range):
        global NOINPUT
        figure = matplotlib.pyplot.figure()
        axes = figure.gca(projection='3d')
        axes.set_xlim3d(axis_range[0], axis_range[1])
        axes.set_ylim3d(axis_range[2], axis_range[3])
        axes.set_zlim3d(axis_range[4], axis_range[5])
        x_points = []
        y_points = []
        z_points = []
        graph_list = []
        for graph_n, line in enumerate(self.point_list):
            line = [point for point in line if self.point_pred(point, axis_range)]
            if len(line) == 0:
                continue
            x_points = [point[0] for point in line]
            y_points = [point[1] for point in line]
            z_points = [point[2] for point in line]
            axes.plot(x_points, y_points, z_points, color="b")

        x_list = [axis_range[0] + (axis_range[1] - axis_range[0])*(i)/100 for i in range(101)]
        y_list = [axis_range[2] + (axis_range[3] - axis_range[2])*(i)/100 for i in range(101)]
        z_list = []
        for y in y_list:
            z_tmp = []
            for x in x_list:
                bx, by, bz = self.field_map.get_field(x, y, axis_range[4])
                btot = (bx**2+by**2+bz**2)**0.5
                if btot > 10.0:
                    btot = 10.0
                z_tmp.append(abs(btot))
            z_list.append(z_tmp)
        print(z_list[0][0:3], min([min(z_tmp) for z_tmp in z_list]), max([max(z_tmp) for z_tmp in z_list]))
        cont = axes.contourf(x_list, y_list, z_list, levels=100, zdir="z", offset=axis_range[5], alpha=0.5)
        figure.colorbar(cont)

        figure2 = matplotlib.pyplot.figure()
        axes2 = figure2.add_subplot(1, 1, 1)
        cont = axes2.contourf(x_list, y_list, z_list, levels=100, zdir="z", offset=axis_range[5], alpha=0.5)


        angle = 45
        dangle = 1
        test = 0
        if input("Press y to spin") == "y":
            test = 1
        while True:
            axes.view_init(30, angle)
            matplotlib.pyplot.draw()
            matplotlib.pyplot.pause(.001)
            if not test:
                break
            if angle == 90 or angle == 0:
                dangle *= -1
            angle += dangle
        input("Press <CR> to end")

def get_field_line_derivative(xvec, t0, args):
    self_field = args
    x, y, z = xvec[0], xvec[1], xvec[2]
    bx, by, bz = self_field.get_field(x, y, z)
    btot = (bx**2+by**2+bz**2)**0.5
    if btot < 1e-3:
        print("Failed to get field at", x, y, z, "return", bx, by, bz, "from field", self_field)
        raise RuntimeError("No field - derivative undefined")
    dx = -bx/btot
    dy = -by/btot
    dz = -bz/btot
    return [dx, dy, dz]

class OpalFieldMap(object):
    def __init__(self, lattice_file):
        PyOpal.parser.initialise_from_opal_file(lattice_file)
    
    def get_field(self, x, y, z):
        oob, bx, by, bz, ex, ey, ez = PyOpal.field.get_field_value(x, y, z, 0.)
        return bx, by, bz

def main():
    os.chdir("output/triplet_baseline/baseline/tmp/find_closed_orbits/")
    field = OpalFieldMap("VerticalSectorFFA.tmp")
    ax_range = [0.0, 5.0, 0.0, 5.0, 0.0, 0.2]
    s_range = [3.0, 5.0, 0.0, 1.0, 0.0, 0.2]
    start_vector = []
    n = 20
    for i in range(n+1):
        for j in range(n+1):
            dx = s_range[1]-s_range[0]
            dy = s_range[3]-s_range[2]
            start_vector.append([s_range[0]+i*dx/n, s_range[2]+j*dy/n, s_range[4]])
    tsteps = [0.01*i for i in range(100)]
    integrator = Integrator3D(field, start_vector, tsteps)
    integrator.get_field_lines()
    integrator.plot_field_line_3d(s_range)

if __name__ == "__main__":
    main()