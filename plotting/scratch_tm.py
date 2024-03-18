import json
import numpy

class TMScratcher:
    def __init__(self):
        self.my_data = []
        self.in_file = "output/2023-03-01_baseline/find_bump_v10/bump=-0.0_by=0.04_bumpp=0.0/find_bump_parameters_003.out"

    def load_data(self):
        self.my_data = []
        fin = open(self.in_file)
        print(f"opened {self.in_file}")
        for line in fin.readlines():
            self.my_data.append(json.loads(line))
        print(f"Loaded {len(self.my_data)} lines")

    def parse_data(self):
        score_elements = self.get_score_elements()
        from_zero_tm_list = []
        for element in score_elements:
            tm = element["transfer_matrix"]
            from_zero_tm_list.append(numpy.array(tm)[0:2, 0:2])
        sequential_tm_list = []
        for i, tm1 in enumerate(from_zero_tm_list[1:]):
            tm0 = from_zero_tm_list[i]
            sequential_tm_list.append(numpy.dot(tm1, numpy.linalg.inv(tm0)))
        print("from zero")
        for matrix in from_zero_tm_list:
            print(matrix)
        print("sequential")
        for matrix in sequential_tm_list:
            print(matrix)
        m = sequential_tm_list[0]
        n = sequential_tm_list[1]
        print("b1 ratio", -n[0,0]*m[0,1]/n[0,1]-m[1,1])
        self.print_params()

    def print_params(self):
        for parameter in self.my_data[-1]["parameters"]:
            print(parameter["name"], parameter["current_value"])

    def get_score_elements(self):
        last_data = self.my_data[-1]
        element_list = []
        station_list = [3, 4, 5]
        for element in last_data["score_list"]:
            if element["score_type"] != "tune":
                continue
            if element["station"] not in station_list:
                continue
            element_list.append(element)
        print("Found", len(element_list), "elements")
        return element_list

def main():
    scratcher = TMScratcher()
    scratcher.load_data()
    scratcher.parse_data()

if __name__ == "__main__":
    main()
