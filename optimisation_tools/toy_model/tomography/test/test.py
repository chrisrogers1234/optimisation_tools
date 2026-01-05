
import math
import bisect

import numpy
import numpy.random
import matplotlib
import matplotlib.pyplot

class TestModel:
    def __init__(self):
        self.radius_squared = [5**2, 10**2] # radial bins, must be sorted
        self.rotation_velocity = [0.1, -0.07] # radians per frame, for each radial bin
        self.n_particles = 10000
        self.n_rows = 10
        self.n_cols = 10
        self.distribution_list = []

    def setup(self):
        self.setup_distribution()

    def setup_distribution(self):
        n_required = self.n_particles
        self.distribution_list.append([])
        while n_required > 0:
            new_particles = numpy.random.multivariate_normal([0.0, 0.0], [[10.0, 0.0], [0.0, 2.0]], n_required).tolist()
            new_particles = [p for p in new_particles if p[0]**2+p[1]**2 < self.radius_squared[-1]]
            self.distribution_list[-1] += new_particles
            n_required = self.n_particles - len(self.distribution_list[-1])
        self.distribution_list[-1] = sorted(self.distribution_list[-1], key = lambda p: p[0]**2+p[1]**2)
        print(f"Distribution had length {len(self.distribution_list[-1])}")

    def do_one_particle(self, particle):
        r2 = particle[0]**2+particle[1]**2
        r_index = bisect.bisect_left(self.radius_squared, r2)
        r_velocity = self.rotation_velocity[r_index]
        new_p = [particle[0]*math.cos(r_velocity)+particle[1]*math.sin(r_velocity),
                -particle[0]*math.sin(r_velocity)+particle[1]*math.cos(r_velocity)]
        return new_p

    def do_one_frame(self):
        new_distribution = [None]*len(self.distribution_list[-1])
        for p_index, particle in enumerate(self.distribution_list[-1]):
            new_particle = self.do_one_particle(particle)
            new_distribution[p_index] = new_particle
        self.distribution_list.append(new_distribution)

    def project_2d(self, turn_number):
        r_max = self.radius_squared[-1]**0.5
        bins = numpy.zeros([self.n_rows, self.n_cols])
        for p in self.distribution_list[turn_number]:
            ix = int((p[0]+r_max)/2/r_max*self.n_cols)
            iy = int((p[1]+r_max)/2/r_max*self.n_rows)
            bins[ix][iy] += 1
        return bins

    def project_1d(self, axis, turn_number):
        """axis is the variable that is kept; so if variable is 0, we keep x (column number) and discard y (row number)"""
        r_max = self.radius_squared[-1]**0.5+1
        n_bins = [self.n_cols, self.n_rows][axis]
        bins = numpy.zeros([n_bins])
        for p in self.distribution_list[turn_number]:
            value = (p[axis]+r_max)/2/r_max*n_bins
            bins[int(value)] += 1
        return bins.tolist()

class PlotTest:
    def __init__(self, output_dir, model):
        self.model = model
        self.output_dir = output_dir

    def plots(self):
        self.plot_mountain()
        self.plot_2d()

    def plot_mountain(self):
        x_axis, y_axis, weights = [], [], []
        n_y = len(self.model.distribution_list)
        n_x = None
        for iy in range(n_y):
            one_d = self.model.project_1d(0, iy)
            y_axis += [iy]*len(one_d)
            x_axis += [ix for ix in range(len(one_d))]
            weights += one_d
            n_x = len(one_d)
        bins_x = numpy.linspace(-0.5, n_x-0.5, n_x+1)
        bins_y = numpy.linspace(-0.5, n_y-0.5, n_y+1)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.hist2d(x_axis, y_axis, bins = [bins_x, bins_y], weights = weights)
        figure.savefig(f"{self.output_dir}/mountain.png")

    def plot_2d(self):
        x_axis, y_axis = [], []
        for ix in range(self.model.n_rows):
            y_axis += [ix]*self.model.n_cols
            x_axis += [iy for iy in range(self.model.n_cols)]
        bins_x = numpy.linspace(-0.5, self.model.n_cols-0.5, self.model.n_cols+1)
        bins_y = numpy.linspace(-0.5, self.model.n_rows-0.5, self.model.n_rows+1)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        for turn_number in range(0, 101, 1):
            weights2d = self.model.project_2d(turn_number)
            weights1d = []
            for row in weights2d.tolist():
                weights1d += row
            axes.clear()
            axes.hist2d(x_axis, y_axis, weights = weights1d, bins=[bins_x, bins_y])
            figure.savefig(f"{self.output_dir}/hist2d_{turn_number+1:03d}.png")

class Tomography:
    def __init__(self):
        pass

def main():
    test_model = TestModel()
    test_model.setup()
    for i in range(100):
        test_model.do_one_frame()
    print("Tracked for 100 frames")
    plotter = PlotTest("output/tomography_test_1/", test_model)
    plotter.plots()
    print(f"Plotted to '{plotter.output_dir}'")

if __name__ == "__main__":
    main()

