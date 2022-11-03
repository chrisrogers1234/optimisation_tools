import random
# curvefit with non linear least squares (curve_fit function)
import numpy
import scipy.optimize
import matplotlib.pyplot as plt

class PolynomialFitter(object):
    def __init__(self, n_dimensions):
        self._arr_d = 0
        self.test_matrix = None
        self.is_linear = True
        self.set_n_dimensions(n_dimensions)

    def set_n_dimensions(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.vec_length = 1+self.n_dimensions
        if self.is_linear:
            return
        for i in range(self.n_dimensions):
            for j in range(i, self.n_dimensions):
                self.vec_length += 1


    def quadratic(self, x_data, matrix):
        y_data = []
        for x_item in x_data:
            y_item = [0.0 for i in range(self.n_dimensions)]
            vector = [1.0]+x_item
            for i, x0 in enumerate(x_item):
                for j, x1 in enumerate(x_item[i:]):
                    vector.append(x0*x1)
            for i, row in enumerate(matrix):
                for j, cell in enumerate(row):
                    y_item[i] += matrix[i][j]*vector[j]
            y_data.append(y_item)
        return y_data

    def linear(self, x_data, matrix):
        y_data = []
        for x_item in x_data:
            y_item = [0.0 for i in range(self.n_dimensions)]
            vector = [1.0]+x_item+[0 for i in range(len(matrix[0])-len(x_item)-1)]
            for i, row in enumerate(matrix):
                for j, cell in enumerate(row):
                    y_item[i] += matrix[i][j]*vector[j]
            y_data.append(y_item)
        return y_data

    def test_data(self):
        n_points = 20
        self.test_matrix = [[random.uniform(0, 1) for i in range(self.vec_length)] for j in range(self.n_dimensions)]
        self.print_array(self.test_matrix)
        x_data = [[random.uniform(0, 1) for j in range(self.n_dimensions)] for i in range(n_points)]
        y_data = self.quadratic(x_data, self.test_matrix)
        for i in range(n_points):
            print(x_data[i], y_data[i])
        return x_data, y_data

    def linear_fit_function(self, x_data, *args):
        test_matrix = [args]
        x_data = x_data.tolist()
        y_item = self.linear(x_data, test_matrix)
        y_item = numpy.array(y_item)
        y_item_0 = y_item[::, 0:1].flatten()

        #print("Fit function on", len(x_data), "points\n", x_data)
        #print("Using test matrix\n", numpy.array(test_matrix))
        #print("Yields\n", y_item_0)

        return y_item_0

    def quad_fit_function(self, x_data, *args):
        test_matrix = [args]
        x_data = x_data.tolist()
        y_item = self.quadratic(x_data, test_matrix)
        y_item = numpy.array(y_item)
        y_item_0 = y_item[::, 0:1].flatten()

        #print("Fit function on", len(x_data), "points\n", x_data)
        #print("Using test matrix\n", numpy.array(test_matrix))
        #print("Yields\n", y_item_0)

        return y_item_0

    def fit_transfer_map(self, x_data, y_data):
        x_data = numpy.array(x_data)
        y_data = numpy.array(y_data)
        max_dim = self.n_dimensions
        parameters_out = []
        if self.is_linear:
            fit_function = self.linear_fit_function
        else:
            fit_function = self.quadratic_fit_function
        for dim in range(max_dim):
            self._arr_d = dim
            y_data_0 = y_data[::, self._arr_d:self._arr_d+1].flatten()
            parameters_in = [1 for i in range(self.vec_length)]
            parameters, parameter_cov = scipy.optimize.curve_fit(fit_function, x_data, y_data_0, parameters_in)
            parameters_out.append(parameters.tolist())
        return parameters_out


    @classmethod
    def print_array(cls, array):
        print(numpy.array2string(numpy.array(array),
              formatter={'float': lambda x: f'{x:.4f}'},
              separator=', '))

def main():
    fitter = PolynomialFitter(4)
    x_data, y_data = fitter.test_data()
    fit = fitter.fit_transfer_map(x_data, y_data)
    print("Ref:")
    PolynomialFitter.print_array(fitter.test_matrix)
    print("Test:")
    PolynomialFitter.print_array(fit)


if __name__ == "__main__":
    main()