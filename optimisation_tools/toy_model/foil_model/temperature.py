import os
import shutil

import numpy
import scipy
import scipy.integrate
import matplotlib
import matplotlib.pyplot

class FoilHeatingModel(object):
    def __init__(self):
        self.number_of_sides = 2
        self.time_step = 0.0
        self.nx = 0
        self.ny = 0
        self.mesh_size = 0.0
        self.conductivity = 0.0
        self.heat_capacity = 0.0
        self.emissivity = 0.0
        self.ambient_temperature_pow4 = 0.0
        self.temperatures = None
        self.thicknesses = None
        self.heating = None

        self.filename = None
        self.foil_hits = None
        self.origin = [0, 0]
        self.n_misses = 0
        self.n_hits = 0

        self.output_list = []
        self.output_times = []

    def setup(self, config):
        for key, value in config.items():
            if key not in self.__dict__:
                raise KeyError("Did not recognise configuration item {0}".format(key))
            self.__dict__[key] = value
        for key in ["temperatures", "thicknesses", "heating"]:
            self.__dict__[key] = numpy.array(self.__dict__[key])
            arr_shape = self.__dict__[key].shape
            if arr_shape != (self.ny, self.nx):
                raise ValueError("{0} shape {3} does not match ({1}, {2})".format(key, self.nx, self.ny, arr_shape))
        return

    def load_foil_hits(self):
        fin = open(self.filename)
        foil_hits = {}
        n_pulses = fin.readline()
        n_pulses = int(n_pulses.split()[1])
        foil_hits["n_pulses"] = n_pulses
        n_per_pulse = fin.readline()
        n_per_pulse = int(n_per_pulse.split()[1])
        foil_hits["n_per_pulse"] = n_per_pulse
        hit_list = []
        for line in fin.readlines():
            hit = line.split()
            hit = [float(hit[0]), float(hit[1])]
            hit_list.append(hit)
        foil_hits["hit_list"] = hit_list
        print("Loaded foil hits...")
        print("    x range:", min([hit[0] for hit in hit_list]), max([hit[0] for hit in hit_list]))
        print("    y range:", min([hit[1] for hit in hit_list]), max([hit[1] for hit in hit_list]))
        self.foil_hits = foil_hits

    def parse_foil_hits(self, dQ, number_actual_protons):
        n_cells = self.nx, self.ny
        heating = [[0 for iy in range(n_cells[1])] for ix in range(n_cells[0])]
        for pos in self.foil_hits["hit_list"]:
            ix = int((pos[0]-self.origin[0])/self.mesh_size)
            iy = int((pos[1]-self.origin[1])/self.mesh_size)
            if ix >= n_cells[0] or ix < 0:
                self.n_misses += 1
                continue
            elif iy >= n_cells[1] or iy < 0:
                self.n_misses += 1
                continue
            #print(pos, ix, iy, len(heating), len(heating[0]), n_cells[0], n_cells[1])
            heating[ix][iy] += 1.0
            self.n_hits += 1
        number_simulated_protons = self.foil_hits["n_pulses"]*self.foil_hits["n_per_pulse"]
        normalisation = dQ*number_actual_protons/number_simulated_protons
        heating = numpy.array([[normalisation*heating[ix][iy] for iy in range(n_cells[1])] for ix in range(n_cells[0])]).transpose()
        self.heating = heating

    def integrate(self):
        self.beam_heating()
        initial_temps = numpy.reshape(self.temperatures, (self.ny*self.nx,) )
        self.output_times = sorted(self.output_times)
        solution = scipy.integrate.solve_ivp(self.derivative, (0.0, self.output_times[-1]), initial_temps, t_eval = self.output_times)
        self.output_list = []
        for row in solution.y.transpose():
            output = numpy.reshape(row, (self.ny, self.nx))
            self.output_list.append(output)
        return self.output_list

    def derivative(self, time, temps):
        self.temperatures = numpy.reshape(temps, self.temperatures.shape)
        derivative_grid = self.conduction_derivative(time)
        derivative_grid += self.radiation_derivative(time)
        derivative_array = numpy.reshape(derivative_grid, (len(temps),))
        return derivative_array

    def conduction_derivative(self, time):
        derivative_grid = numpy.ndarray((self.ny, self.nx))
        for ix in range(self.nx):
            for iy in range(self.ny):
                heat_transfer = 0.0
                for direction in [-1, 1]:
                    jx, jy = ix+direction, iy
                    heat_transfer += self.heat_transfer(ix, iy, jx, jy) # heat transfer from ix, iy to jx, jy
                for dy, direction in [(-1, -1), (0, 1)]:
                    jx, jy = ix, iy+direction
                    heat_transfer += self.heat_transfer(ix, iy, jx, jy) # heat transfer from ix, iy to jx, jy [J]
                # temperature change in cell[ix, iy] [K]
                derivative_grid[iy, ix] = heat_transfer/(self.heat_capacity*self.volume(ix, iy)) 
        return derivative_grid

    def heat_transfer(self, ix, iy, jx, jy):
        """Calculate the heat transfer from cell [ix, iy] to cell [jx, jy]"""
        if ix < 0 or iy < 0 or jx < 0 or jy < 0:
            return 0.0
        if ix >= self.nx or iy >= self.ny or jx >= self.nx or jy >= self.ny:
            return 0.0
        temp_gradient = -(self.temperatures[iy, ix]-self.temperatures[jy, jx])/self.mesh_size
        thickness = min(self.thicknesses[iy, ix], self.thicknesses[jy, jx])
        heat_flow_rate = self.conductivity*thickness*self.mesh_size*temp_gradient
        return heat_flow_rate

    def volume(self, ix, iy):
        """Return the volume of cell[ix, iy] in mm^3"""
        return self.thicknesses[iy, ix]*self.mesh_size**2

    def radiation_derivative(self, time):
        derivative_grid = numpy.ndarray((self.ny, self.nx))
        const = -self.number_of_sides*self.mesh_size**2*self.stefan_boltzmann*self.emissivity
        for ix in range(self.nx):
            for iy in range(self.ny):
                temp_pow4 = self.temperatures[iy, ix]**4
                dW = const*(temp_pow4 - self.ambient_temperature_pow4)
                derivative_grid[iy, ix] = dW/(self.heat_capacity*self.volume(ix, iy))
        return derivative_grid

    def beam_heating(self):
        """
        Before integration starts, we give the foil a temperature kick (representing a passing beam)
        """
        for ix in range(self.nx):
            for iy in range(self.ny):
                self.temperatures[iy, ix] += self.heating[iy, ix]/(self.heat_capacity*self.volume(ix, iy))


    stefan_boltzmann = 5.670373e-8*1e-6*1e-9 # stefan_boltzmann constant [J/ns/mm^2/K^4]

def config_sns():
    nx = 18
    ny = 21
    dQ = 1.9*260e-6*1.602176634e-19*1e6 # [J] ... 1.9 MeV cm^2 / g * 260 mu g / cm^2; energy deposit per proton
    config = {
        "mesh_size":1.0, # mm
        "time_step":1e3, # ns
        "conductivity":129.0e-12, # J/ns/mm/K, big range really
        "heat_capacity":1.534*1e-3, # J/mm^3/K
        "emissivity":0.80,
        "ambient_temperature_pow4":293**4, # K^4
        "nx":nx,
        "ny":ny,
        "output_times":[i * 1e9/60/20 for i in range(21)], # ns
        "thicknesses":[[1.0e-3 for i in range(nx)] for j in range(ny)], # mm
        "temperatures":[[293.0 for i in range(nx)] for j in range(ny)], # K
        "heating":[[0.0 for i in range(nx)] for j in range(ny)], # J/repetition
        "number_of_sides":2,
    }
    for ix in range(8, 12):
        for iy in range(8, 12):
            #config["temperatures"][iy][ix] = 1000.0
            config["temperatures"][iy][ix] = 859.0
            config["heating"][iy][ix] = dQ*2e13
    return config


def config_fets_ring():
    nx = 18*4-1
    ny = 200
    config = {
        "filename":"output/2023-03-01_baseline/toy_model_painting_v26/corr=True/foil_hits.txt",
        "origin":[-16.0, -25.0], # position of 0,0 in the grid, mm
        "mesh_size":0.25, # mm
        "time_step":1e3, # ns
        "conductivity":129.0e-12, # J/ns/mm/K, big range really
        "heat_capacity":1.534*1e-3, # J/mm^3/K
        "emissivity":0.80,
        "ambient_temperature_pow4":293**4, # K^4
        "nx":nx,
        "ny":ny,
        "output_times":[i * 1e9/1000 for i in range(11)], # ns
        "thicknesses":[[100.0e-6 for i in range(nx)] for j in range(ny)], # mm
        "temperatures":[[293.0 for i in range(nx)] for j in range(ny)], # K
        "heating":[[0.0 for i in range(nx)] for j in range(ny)], # J/repetition
        "number_of_sides":2,
    }
    #for ix in range(8, 12):
    #    for iy in range(8, 12):
    #        config["heating"][iy][ix] = dQ*1e11/16.0
    return config


fig_index = 0
def plot_foil_temperature(output_dir, heating, step):
    global fig_index
    fig_index += 1
    solution = heating.output_list[step].transpose()
    nx, ny = solution.shape
    x_list = []
    y_list = []
    w_list = []
    bins = [
        [(ix-0.5)*heating.mesh_size+heating.origin[0] for ix in range(nx+1)],
        [(iy-0.5)*heating.mesh_size+heating.origin[1] for iy in range(ny+1)]
    ]
    for ix in range(nx):
        for iy in range(ny):
            x_list.append(ix*heating.mesh_size+heating.origin[0])
            y_list.append(iy*heating.mesh_size+heating.origin[1])
            w_list.append(solution[ix, iy])
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    h = axes.hist2d(x_list, y_list, bins=bins, weights=w_list, vmin=250.0, vmax=500.0)
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    figure.colorbar(h[3])
    figure.suptitle("Max T: {0:.4g} K at time: {1:.4g} ms".format(numpy.amax(solution), heating.output_times[step]/1e6))
    figure.savefig("{1}/foil_temp_{0:04d}.png".format(fig_index, output_dir))
    if fig_index > 5:
        matplotlib.pyplot.close(figure)

def plot_temperature_vs_time(output_dir, time_list, mean_temp_list, max_temp_list):
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.plot([t*1e-6 for t in time_list], mean_temp_list)
    axes.plot([t*1e-6 for t in time_list], max_temp_list)
    axes.set_xlabel("Time [ms]")
    axes.set_ylabel("Temperature [K]")
    figure.savefig(f"{output_dir}/foil_temperature_vs_time.png")


def main():
    output_dir = "output/2023-03-01_baseline/foil_heating_test_fets_ring_corr=True_v2/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    heating = FoilHeatingModel()
    heating.setup(config_fets_ring())
    if heating.filename:
        dQ = 105*20e-6*1.602176634e-19*1e6 # [J] ... 105 MeV cm^2 / g * 20 mu g / cm^2; energy deposit per proton
        heating.load_foil_hits()
        heating.parse_foil_hits(dQ, 3e11)

    output_times = [0.0]
    mean_temp_list, max_temp_list = [], []
    for i in range(10):
        solution = heating.integrate()
        output_times += [t+output_times[-1] for t in heating.output_times]
        for i, row in enumerate(solution):
            mean_temp_list.append(numpy.mean(row))
            max_temp_list.append(numpy.amax(row))
            print(f"Foil misses:{heating.n_misses}/{heating.n_hits+heating.n_misses} mean: {numpy.mean(row):.4g} K max: {numpy.amax(row):.4g} K at time {heating.output_times[i]:.4g} ns\n")
            plot_foil_temperature(output_dir, heating, i)
    output_times = output_times[1:]
    plot_temperature_vs_time(output_dir, output_times, mean_temp_list, max_temp_list)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")