import os
import matplotlib.pyplot

class PlotG4Fields(object):
    def __init__(self):
        self.data = []
        self.output_dir = ""

    def load_file(self, file_name):
        self.data = []
        print("opening", file_name)
        fin = open(file_name)
        line = "new"
        while line:
            line = fin.readline()
            try:
                values = [float(word) for word in line.split()]
            except (IndexError, ValueError):
                continue
            if len(values) == len(self.labels):
                self.data.append(values)

    def plot(self, axes, x_index, y_index):
        x_values = [values[x_index] for values in self.data]
        y_values = [values[y_index] for values in self.data]
        axes.plot(x_values, y_values, label = self.labels[y_index])
        axes.set_xlabel(self.labels[x_index])
        axes.set_ylabel(self.labels[y_index])

    def plot_bfield(self, z_lim=None):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        self.plot(axes, 2, 3)
        self.plot(axes, 2, 4)
        self.plot(axes, 2, 5)
        axes.set_ylabel("B [T]")
        axes.set_ylim([-0.5, 0.5])
        axes.legend()
        if z_lim:
            axes.set_xlim(z_lim)
        figure.savefig(os.path.join(self.output_dir, "bfield.png"))

    #x y z t Bx By Bz Ex Ey Ez
    labels = ["x [mm]", "y [mm]", "z [mm]", "t [ns]",
              "B$_x$ [T]", "B$_y$ [T]", "B$_z$ [T]", 
              "E$_x$ [MV/m]", "E$_y$ [MV/m]", "E$_z$ [MV/m]", 
            ]

def main():
    output_dir = "output/ruihu_cooling_v1/test/"
    file_name = os.path.join(output_dir, "tmp/find_closed_orbits/field_cell.dat")
    plotter = PlotG4Fields()
    plotter.output_dir = output_dir
    plotter.labels = plotter.labels[0:3] +plotter.labels[4:7]
    plotter.load_file(file_name)
    plotter.plot_bfield()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")