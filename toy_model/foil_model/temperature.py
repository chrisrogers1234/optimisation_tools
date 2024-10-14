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
        self.output_times = []
        self.temperatures = None
        self.thicknesses = None
        self.heating = None

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

    def integrate(self):
        self.beam_heating()
        initial_temps = numpy.reshape(self.temperatures, (self.ny*self.nx,) )
        self.output_times = sorted(self.output_times)
        solution = scipy.integrate.solve_ivp(self.derivative, (0.0, self.output_times[-1]), initial_temps, t_eval = self.output_times)
        output_list = []
        for row in solution.y.transpose():
            output = numpy.reshape(row, (self.ny, self.nx))
            output_list.append(output)
        return output_list

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
    nx = 20
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
    nx = 20
    ny = 21
    dQ = 105*20e-6*1.602176634e-19*1e6 # [J] ... 105 MeV cm^2 / g * 20 mu g / cm^2; energy deposit per proton
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
        "thicknesses":[[100.0e-6 for i in range(nx)] for j in range(ny)], # mm
        "temperatures":[[293.0 for i in range(nx)] for j in range(ny)], # K
        "heating":[[0.0 for i in range(nx)] for j in range(ny)], # J/repetition
        "number_of_sides":2,
    }
    for ix in range(8, 12):
        for iy in range(8, 12):
            #config["temperatures"][iy][ix] = 1000.0
            #config["temperatures"][iy][ix] = 859.0
            config["heating"][iy][ix] = dQ*1e11/16.0
    return config


fig_index = 0
def plot_foil_temperature(output_dir, solution):
    global fig_index
    fig_index += 1
    nx, ny = solution.shape
    x_list = []
    y_list = []
    w_list = []
    for ix in range(nx):
        for iy in range(ny):
            x_list.append(ix+0.5)
            y_list.append(iy+0.5)
            w_list.append(solution[ix, iy])
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    h = axes.hist2d(x_list, y_list, bins=[range(nx+1), range(ny+1)], weights=w_list, vmin=0.0, vmax=2000.0)
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    figure.colorbar(h[3])
    figure.suptitle("Max T: {0:.4g} K".format(numpy.amax(solution)))
    figure.savefig("{1}/foil_temp_{0}.png".format(fig_index, output_dir))
    if fig_index > 5:
        matplotlib.pyplot.close(figure)

def main():
    output_dir = "output/foil_heating_test_fets_ring"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    heating = FoilHeatingModel()
    heating.setup(config_sns())
    for i in range(10):
        solution = heating.integrate()
        for i, row in enumerate(solution):
            print("{0} mean: {1:.4g} K max: {2:.4g} K at time {3:.4g} ns\n".format("row", numpy.mean(row), numpy.amax(row), heating.output_times[i]))
            plot_foil_temperature(output_dir, row)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")