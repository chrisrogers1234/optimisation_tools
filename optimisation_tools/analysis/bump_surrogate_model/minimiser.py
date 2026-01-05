import copy
import ROOT

def get_point(map_, point_seed_list, value, limits):
    minimiser = Minimiser(map_, point_seed_list, value, limits)
    minimiser.run_minuit()
    return minimiser.point

def get_multipoint(map_, point_seed_list, value, limits):
    minimiser = Minimiser(map_, point_seed_list, value, limits)
    minimiser.run_multi_minuit()
    return minimiser.multipoint


class Minimiser(object):
    def __init__(self, map_, point_seed_list, value, limits):
        self.map = map_
        self.point_list = point_seed_list
        self.ref_value = value
        self.score = [tol*10 for tol in self.tolerance]
        self.point = self.point_list[0]
        self.limits = limits
        self.multipoint = []
        #self.print(test_value)

    def setup_minuit(self):
        if self.minuit == None:
            self.minuit = ROOT.TMinuit(4)
            self.minuit.SetPrintLevel(-1)
        for i in range(4):
            self.minuit.DefineParameter(i, str(i), self.point[i], 0.01,
                                    min(self.limits[i]), max(self.limits[i]))
        self.minuit.SetFCN(self.get_score)
        self.minuit.Command("MIGRAD 500 1.e-6")

    def run_minuit(self):
        for self.point in reversed(self.point_list):
            self.setup_minuit()
            if self.in_tolerance():
                #print("In tolerance", self.map.function(self.point), self.ref_value)
                return
        print("Out of tolerance", self.score, self.tolerance, len(self.point_list))
        raise ValueError("Out of tolerance")
        self.point = [0., 0., 0., 0.]
        return

    def run_multi_minuit(self):
        for self.point in reversed(self.point_list):
            self.setup_minuit()
            if self.in_tolerance():
                #print("In tolerance", self.map.function(self.point), self.ref_value)
                print("+", end="")
                self.multipoint.append(copy.deepcopy(self.point))
            else:
                print(".", end="")
        print()

    def in_tolerance(self):
        for i in range(4):
            if abs(self.score[i]) > self.tolerance[i]:
                return False
        return True


    def get_score(self, nvar, parameters, score, jacobian, err):
        for i in range(4):
            var = ROOT.Double()
            err = ROOT.Double()
            self.minuit.GetParameter(i, var, err)
            self.point[i] = float(var)
        test_value = self.map.function(self.point)
        #self.print(test_value)
        self.score = [(test_value[i] - self.ref_value[i])**2 for i in range(4)]
        score[0] = sum(self.score)

    def print(self, test_value):
        print("point: ", end="")
        for i in range(4):
            print(format(self.point[i], "10.6g"), end="")
        print(" ref val: ", end="")
        for i in range(4):
            print(format(self.ref_value[i], "10.6g"), end="")
        print(" test val: ", end="")
        for i in range(4):
            print(format(test_value[i], "10.6g"), end="")
        print()

    tolerance = [0.001]*4
    minuit = None
