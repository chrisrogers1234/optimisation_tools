
class MonitorDigitisation():
    def __init__(self):
        self.t_bins = [0.0]
        self.t_hist = [0]
        self.t_resolution = 10.0
        self.file_name = "./monitor.dat"

    def append_time(self, time):
        t_bin = int((time-self.t_bins[0])/self.t_resolution)
        if t_bin < 0:
            delta = abs(t_bin)
            self.t_bins = [self.t_bins[0]-self.t_resolution*(i+1) for i in reversed(range(delta+1))]+self.t_bins
            self.t_hist = [0 for i in range(delta+1)] + self.t_hist
            t_bin += delta
        elif t_bin >= len(self.t_hist):
            delta = t_bin - len(self.t_bins)
            self.t_bins = self.t_bins + [self.t_bins[-1]+self.t_resolution*(i+1) for i in range(delta+1)]
            self.t_hist = self.t_hist + [0 for i in range(delta+1)]
        self.t_hist[t_bin] += 1

    def do_one_turn(self, turn, particle_collection):
        for p in particle_collection:
            self.append_time(p.t)

    def load(self):
        with open(self.file_name) as fin:
            data = [line.rstrip().split() for line in fin.readlines()]
        self.t_bins = [float(item[0]) for item in data]
        self.t_hist = [int(item[1]) for item in data]
        self.t_bins.append(2*self.t_bins[-1]-self.t_bins[-2])

    def save(self):
        with open(self.file_name, "w") as fout:
            for i in range(len(self.t_hist)):
                fout.write(f"{self.t_bins[i]} {self.t_hist[i]} \n")

def test_monitor():
    monitor_dig = MonitorDigitisation()
    for t in [1, 2, 3, 4, 5]:
        monitor_dig.append_time(t)
    print([0.0, 10.0], monitor_dig.t_bins)
    print([5],  monitor_dig.t_hist)

    for t in [10.0, 10.1, 19.9, 20.0]:
        monitor_dig.append_time(t)
    print([0.0, 10.0, 20.0], monitor_dig.t_bins)
    print([5, 3, 1],  monitor_dig.t_hist)

    for t in [71.0, 65.1, 74.9, 62.0]:
        monitor_dig.append_time(t)
    print([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], monitor_dig.t_bins)
    print([5,      3,    1,    0,    0,    0,    2,    2], monitor_dig.t_hist)

    for t in [-21.0, -15.1, -24.9, -12.0]:
        monitor_dig.append_time(t)
    print([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], monitor_dig.t_bins)
    print([    2,     2,     0,   5,    3,    1,    0,    0,    0,    2,    2], monitor_dig.t_hist)
