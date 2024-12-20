import bisect
import os

import scipy
import matplotlib
import matplotlib.pyplot


class MonitorAnalysis():
    def __init__(self):
        self.dt = 50000
        self.output_directory = "./"
        self.monitor = None
        self.model = None

    def do_plot(self):
        self.oned_plot()
        self.mountain_plot()
        self.all_fft_plots()

    def oned_plot(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.bar(self.monitor.t_bins[:-1], self.monitor.t_hist, self.monitor.t_resolution)
        axes.set_title("Beam Monitor")
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("N")
        self.model.plot_rf_data(axes, 100)
        figure.savefig(os.path.join(self.output_directory, "monitor.png"))
        return

    def t_list(self):
        t_list = [0.0]
        while t_list[-1] < self.monitor.t_bins[-1]:
            period = self.model.harmonic_number/self.model.rf_program.get_frequency(t_list[-1])
            t_list.append(t_list[-1]+period)
        t_index = 0
        x_values, y_values = [], []
        for t in self.monitor.t_bins[:-1]:
            if t > t_list[t_index+1]:
                t_index += 1
            x_values.append(t - t_list[t_index])
            y_values.append(t_index)
        n_x = int(max(x_values)/self.monitor.t_resolution)+1
        x_bins = [i*self.monitor.t_resolution for i in range(n_x)]
        y_bins = [-0.5]+[i+0.5 for i in range(t_index)]
        return x_values, y_values, x_bins, y_bins

    def mountain_plot(self):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        x_values, y_values, x_bins, y_bins = self.t_list()
        axes.hist2d(x_values, y_values, [x_bins, y_bins], weights=self.monitor.t_hist)
        axes.set_title("Beam Monitor")
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("RF period $\\times$ harmonic number")
        self.model.plot_rf_data(axes, 100)
        figure.savefig(os.path.join(self.output_directory, "monitor_mountain.png"))
        return

    def all_fft_plots(self):
        try:
            for i, t0 in enumerate(self.model.rf_program.t_list[:-1]):
                t1 = self.model.rf_program.t_list[i+1]
                f0 = 0.1*self.model.rf_program.get_frequency(t0)
                f1 = 10*self.model.rf_program.get_frequency(t0)
                self.fft_plot(t0, t1, f0, f1)
        except:
            print("No t_list in rf_program - skipping")
        f0 = 0.5*self.model.rf_program.get_frequency(t0)
        f1 = 10*self.model.rf_program.get_frequency(t0)
        self.fft_plot(self.monitor.t_bins[0], self.monitor.t_bins[-1], f0, f1)

    def fft_plot(self, t0, t1, f0, f1):
        i0 = bisect.bisect_left(self.monitor.t_bins, t0)
        i1 = bisect.bisect_left(self.monitor.t_bins, t1)
        if i1 >= len(self.monitor.t_bins):
            i1 = -1
        fft_out = scipy.fft.fft(self.monitor.t_bins[i0:i1])
        fft_freq = [i/(t1-t0) for i in range(i1-i0)]
        i0 = bisect.bisect_left(fft_freq, f0)
        i1 = bisect.bisect_left(fft_freq, f1)
        #df = 1/(t1-t0)
        #f_max = 1/self.monitor.t_resolution
        figure = matplotlib.pyplot.figure(figsize=(20,10))
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(fft_freq[i0:i1], fft_out.real[i0:i1])#, fft_freq[i0:i1], fft_out.imag[i0:i1])
        f_rf = [self.model.rf_program.get_frequency(t0)]*2
        ylim = axes.get_ylim()
        axes.plot(f_rf, ylim, linestyle="dashed", color="grey")
        axes.set_ylim(ylim)
        axes.text(0.01, 0.95, f"Time window: {t0:.1f} {t1:.1f} [ns]",
                  transform=axes.transAxes)
        axes.set_xlabel("Frequency [GHz]")
        figure.savefig(os.path.join(self.output_directory, f"fft_{t0:.1f}_{t1:.1f}.png"))
