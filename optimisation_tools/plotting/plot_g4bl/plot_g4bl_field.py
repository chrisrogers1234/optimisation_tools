import os
import matplotlib.pyplot

class PlotG4Fields(object):
    def __init__(self):
        self.data = {}
        self.output_dir = ""

    def load_file(self, file_name, label_name=""):
        self.data[label_name] = []
        print("opening", label_name, file_name)
        fin = open(file_name)
        line = "new"
        while line:
            line = fin.readline()
            try:
                values = [float(word) for word in line.split()]
            except (IndexError, ValueError):
                continue
            if len(values) == len(self.labels):
                self.data[label_name].append(values)

    def plot(self, axes, x_index, y_index):
        for key in self.data:
            prefix = ""
            if key:
                prefix = key+": "
            x_values = [values[x_index] for values in self.data[key]]
            y_values = [values[y_index] for values in self.data[key]]
            axes.plot(x_values, y_values, label = prefix+self.labels[y_index])
            axes.set_xlabel(self.labels[x_index])
            axes.set_ylabel(self.labels[y_index])

    def plot_bfield(self, z_lim=None):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)
        self.plot(axes, 2, 6)
        axes.set_ylabel("B [T]")
        axes.set_ylim([-10.0, 10.0])
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
    output_dir = "output/demo_apr25_v5/"
    file_name = os.path.join(output_dir, "pz_beam=150/tmp/find_closed_orbits/fieldmap.dat.txt")
    plotter = PlotG4Fields()
    plotter.output_dir = output_dir
    plotter.load_file(file_name, "harmonic")
    plotter.load_file(file_name.replace("v5", "v3"), "sheets")
    plotter.plot_bfield()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")