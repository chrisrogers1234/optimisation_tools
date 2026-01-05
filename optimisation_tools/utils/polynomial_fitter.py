import random
# curvefit with non linear least squares (curve_fit function)
import numpy
import scipy.optimize
import matplotlib.pyplot as plt



class PolynomialFitter(object):
    """
    Try to find polynomial such that

    y = y_0 + M_2 x + M_3 x * x

    where
    * y and x are vectors of dimension N
    * M_2 is a matrix of dimension N x N
    * M_3 is a tensor of dimension N x N x N

    In order to do the fitting, we use a fit vector like
    v_0 = [m_0, m2_00 x_0, m2_01 x_1, ..., m3_000 x_0*x_0, m3_001 x_0*x_1, ...]

    v_i = [m_i, ..., m2_ij x_j,  ..., m3_ijk x_j x_k, ...]

    The aim is to minimise the m values so that y is as close as possible to the
    inputs for given x inputs.

    In accelerator jargon, M_2 is the transfer matrix and y_0 is the closed orbit.
    M_3 is a quadratic correction factor.


    If self.is_linear is true, m3_ijk are all assumed to be 0
    """


    def __init__(self, n_dimensions):
        """
        Initialise the fitter
        """
        self.is_linear = True
        self.set_n_dimensions(n_dimensions)

        # just used for testing (should probably be somewhere else, never mind)
        self.test_matrix = None

    def set_n_dimensions(self, n_dimensions):
        """
        Set the dimension

        Here we just set the length of v for convenience later on
        """
        self.n_dimensions = n_dimensions
        self.vec_length = 1+self.n_dimensions
        if self.is_linear:
            return
        for i in range(self.n_dimensions):
            for j in range(i, self.n_dimensions):
                self.vec_length += 1


    def quadratic(self, x_data, matrix):
        """
        Calculate the v vector assuming quadratic expansion
        """
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
        """
        Calculate the v vector assuming linear expansion
        """
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
        """
        Generate some toy data for testing based on test_matrix
        """
        n_points = 20
        self.test_matrix = [[random.uniform(0, 1) for i in range(self.vec_length)] for j in range(self.n_dimensions)]
        self.print_array(self.test_matrix)
        x_data = [[random.uniform(0, 1) for j in range(self.n_dimensions)] for i in range(n_points)]
        y_data = self.quadratic(x_data, self.test_matrix)
        for i in range(n_points):
            print(x_data[i], y_data[i])
        return x_data, y_data

    def linear_fit_function(self, x_data, *args):
        """
        Fit using linear expansion
        """
        test_matrix = [args]
        x_data = x_data.tolist()
        y_item = self.linear(x_data, test_matrix)
        y_item = numpy.array(y_item)
        y_item_0 = y_item[::, 0:1].flatten()
        return y_item_0

    def quad_fit_function(self, x_data, *args):
        """
        Fit using quadratic expansion
        """
        test_matrix = [args]
        x_data = x_data.tolist()
        y_item = self.quadratic(x_data, test_matrix)
        y_item = numpy.array(y_item)
        y_item_0 = y_item[::, 0:1].flatten()
        return y_item_0

    def fit_transfer_map(self, x_data, y_data):
        """
        Do the fit.

        I use scipy.optimize.curve_fit. In principle it is a linear problem,
        so one could use linear least squares, but this way we can get an
        improvement if there is noise and the problem is overconstrained.
        """
        x_data = numpy.array(x_data)
        y_data = numpy.array(y_data)
        max_dim = self.n_dimensions
        parameters_out = []
        if self.is_linear:
            fit_function = self.linear_fit_function
        else:
            fit_function = self.quadratic_fit_function
        for dim in range(max_dim):
            _arr_d = dim
            y_data_0 = y_data[::, _arr_d:_arr_d+1].flatten()
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
    """
    This runs a simple test
    """
    fitter = PolynomialFitter(4)
    x_data, y_data = fitter.test_data()
    fit = fitter.fit_transfer_map(x_data, y_data)
    print("Ref:")
    PolynomialFitter.print_array(fitter.test_matrix)
    print("Test:")
    PolynomialFitter.print_array(fit)


if __name__ == "__main__":
    main()