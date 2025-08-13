import copy
import sys
import bisect
import os
import json

import numpy
import scipy
import matplotlib
import matplotlib.pyplot

class DummyCT():
    def __init__(self):
        pass

    def fitted(self, t):
        return 1.0


class MonitorAnalysis():
    def __init__(self):
        self.dt = 50000
        self.output_directory = "./"
        self.monitor = None
        self.model = None
        self.ct = DummyCT() # current transformer
        self.rf_data = None
        self.max_oned_histo_size = 750000
        self.file_prefix = ""
        self.mountain_chunk_size = 500
        self.fit_stroke = 10
        self.fit_lambda = 1e16 # 0 is perfect spline, higher values weights for second derivative
        self.integral_fit_function = None
        self.mountain_offset = None
        self.mountain_hist = None
        self.problem_title = ""

    def do_plot(self):
        print("Monitor analysis")
        print("  oned plot")
        #self.oned_plot()
        print("  Mountain plot")
        self.mountain_plot()
        print("  Integral plot")
        self.integral_plot()
        print("  Derivative plot")
        self.derivative_plot()
        print("  FFT plots")
        self.all_fft_plots()

    def oned_plot(self):
        n_histos = int(len(self.monitor.t_hist)/self.max_oned_histo_size)+1
        n_points = int(len(self.monitor.t_hist)/n_histos)+1
        print(f"    Doing {n_histos} histograms")
        for i in range(n_histos):
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1, 1, 1)
            n_down, n_up = i*n_points, (i+1)*n_points
            t_bins = self.monitor.t_bins[n_down:n_up]
            t_hist = self.monitor.t_hist[n_down:n_up]
            if len(t_bins) != len(t_hist):
                t_bins = t_bins[:len(t_hist)]
            print(n_down, n_up) #".", end="")
            axes.bar(t_hist, t_bins, self.monitor.t_resolution)
            axes.set_title("Beam Monitor")
            axes.set_xlabel("time [ns]")
            axes.set_ylabel("N")
            axes.text(0.01, 0.95, self.problem_title,
                      transform=axes.transAxes)
            figure.savefig(os.path.join(self.output_directory, f"{self.file_prefix}monitor_{i}.png"))
            matplotlib.pyplot.close(figure)
        print(" done")
        return

    def get_period(self, time):
        """Bug: assumes constant frequency"""
        period = self.model.harmonic_number/self.model.rf_program.get_frequency(time)
        return period

    def t_list(self):
        t_list = [0.0]
        while t_list[-1] < self.monitor.t_bins[-1]:
            t_list.append(t_list[-1]+self.get_period(t_list[-1]))
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

    def offset_mountain_plot(self, x_values, y_values, x_bins, y_bins, weights):
        """
        Find the minimum i.e. gap between two buckets and offset data

        Subtract the minimum value from each row
        """
        a_hist = numpy.histogram2d(x_values, y_values, [x_bins, y_bins], weights=self.monitor.t_hist)[0]
        print("At start x:", len(x_values), "y:", len(y_values), "z:", len(self.monitor.t_hist), "xb:", len(x_bins), "yb:", len(y_bins), "nb:", (len(x_bins)-1)*(len(y_bins)-1))
        a_hist_trans = a_hist.transpose()
        a_hist_sum = [sum(bins) for bins in a_hist] # assume the minimum is at the point where the sum over the entire column is smallest
        offset = a_hist.shape[0] - a_hist_sum.index(min(a_hist_sum))
        print("HISTOGRAM shape", a_hist.shape, "offset", offset, "x[offset]", x_bins[-1] - x_bins[offset])
        # pad the weights at the *beginning* so that the first real weight is
        # at weights[offset]
        # pad with minimum value from the row; for some reason output is column, row
        min_value = min(a_hist_trans[0])
        weights = numpy.append([min_value]*offset, self.monitor.t_hist)
        # pad the coordinates at the *end* so that the coordinates of the old data
        # is updated
        x_values = x_values+x_values[0:offset]
        y_values = y_values+[y_values[-1]+1]*offset
        weights_raw = copy.deepcopy(weights)
        # find the minimum in each row and subtract
        turn_number_list, offset_list = [], []
        for col_index, row in enumerate(a_hist_trans):
            a_min = min(row)
            offset_list.append(a_min)
            turn_number_list.append(col_index)
            for index in range(col_index*len(row), (col_index+1)*len(row)):
                try:
                    weights[index] -= a_min
                except IndexError:
                    break
        # Histogram of number of counts having voltage X at each turn
        # Graph of minimum voltage vs turn number
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        turn_number_list_2d = []
        turn_number = 0
        while len(turn_number_list_2d) < len(weights):
            turn_number_list_2d += [turn_number]*(len(x_bins)-1)
            turn_number += 1
        turn_number_list_2d = turn_number_list_2d[:len(weights)]
        print("Voltages:", len(weights), "turn numbers:", len(turn_number_list_2d))
        h2d = axes.hist2d(turn_number_list_2d, weights_raw, bins=[len(y_bins)-1, 50])
        #offset_alt_list = self.get_min_list(h2d)
        #turn_number_list = [i for i, x in enumerate(offset_alt_list)]
        #axes.scatter(turn_number_list, offset_list, s=1)
        axes.set_xlabel("turn number")
        axes.set_ylabel("Voltage offset [V]")
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        figure.savefig(os.path.join(self.output_directory, f"{self.file_prefix}monitor_voltage_hist_by_turn.png"))
        figure.clear()
        matplotlib.pyplot.close(figure)

        print("At end x:", len(x_values), "y:", len(y_values), "z:", len(weights), "xb:", len(x_bins), "yb:", len(y_bins), "nb:", (len(x_bins)-1)*(len(y_bins)-1))
        return x_values, y_values, x_bins, y_bins, weights

    def get_min_list(self, h2d):
        h2d_data = h2d[0]
        voltage_bins = h2d[2]
        offset_alt_list = []
        for column in h2d_data:
            most_likely_bins = numpy.where(column == max(column))
            most_likely_voltages = [(voltage_bins[i]+voltage_bins[i+1])/2 for i in most_likely_bins]
            mean_voltage = sum(most_likely_voltages)/len(most_likely_voltages)
            offset_alt_list.append(mean_voltage[0])
        return offset_alt_list


    def mountain_plot(self):
        #self.model.plot_rf_data(axes, 100)
        x_values, y_values, x_bins, y_bins = self.t_list()
        x_values, y_values, x_bins, y_bins, weights = self.offset_mountain_plot(x_values, y_values, x_bins, y_bins, weights=self.monitor.t_hist)
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        self.mountain_hist = axes.hist2d(x_values, y_values, [x_bins, y_bins], weights=weights)[0]
        axes.set_title("Beam Monitor")
        axes.set_xlabel("time [ns]")
        axes.set_ylabel("time/(RF period $\\times$ harmonic number)")
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        figure.savefig(os.path.join(self.output_directory, f"{self.file_prefix}monitor_mountain.png"))
        ylim = axes.get_ylim()
        new_ylim = ylim[0], ylim[0]+self.mountain_chunk_size
        while new_ylim[0] < ylim[1]:
            axes.set_ylim(new_ylim)
            figure.savefig(os.path.join(self.output_directory, f"{self.file_prefix}monitor_mountain_{new_ylim[0]}.png"))
            new_ylim = (new_ylim[0]+self.mountain_chunk_size, new_ylim[1]+self.mountain_chunk_size)
        figure.clear()
        matplotlib.pyplot.close(figure)

    def integral_plot(self):
        figure_sum = matplotlib.pyplot.figure()
        axes = figure_sum.add_subplot(1, 1, 1)
        row_sum = [sum(row) for row in self.mountain_hist.transpose()]
        period = self.get_period(0.0)
        time_list = [i*period for i in range(len(row_sum))]
        norm_row_sum = [a_sum/self.ct.fitted(time_list[i]) for i, a_sum in enumerate(row_sum)]
        self.integral_fit_function = scipy.interpolate.make_smoothing_spline(
                time_list[::self.fit_stroke], norm_row_sum[::self.fit_stroke],
                lam=self.fit_lambda)
        self.process_monitor_data()

        axes.plot([self.capture_dict["max_time"], self.capture_dict["max_time"]], [0, 10], linestyle="--", color="grey")
        axes.plot([min(time_list), max(time_list)], [self.capture_dict["max_value"], self.capture_dict["max_value"]], linestyle="--", color="grey")

        axes.plot(time_list, row_sum, label="Integral")
        axes.plot(time_list, norm_row_sum, label="Integral normalised to CT")
        fit_sum = [self.integral_fit_function(t) for t in time_list]
        axes.plot(time_list, fit_sum, label="Fitted data")
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Integrated beam monitor signal [AU]")
        axes.set_ylim([-10, 10])
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        axes.legend()


        figure_sum.savefig(os.path.join(self.output_directory, f"{self.file_prefix}monitor_turn_integral.png"))
        figure_sum.clear()
        matplotlib.pyplot.close(figure_sum)

    def derivative_plot(self):
        n_points = 1000
        figure_sum = matplotlib.pyplot.figure()
        axes = figure_sum.add_subplot(1, 1, 1)
        t_range = [self.monitor.t_bins[0], self.monitor.t_bins[-1]]
        time_list = numpy.linspace(t_range[0], t_range[1], n_points+1)
        point_list = [self.integral_fit_function(t) for t in time_list]
        dt_list = []
        derivative_list = []
        for i in range(n_points):
            dt = (time_list[i]+time_list[i+1])/2.0
            derivative = (point_list[i]-point_list[i+1])/(time_list[i]-time_list[i+1])
            dt_list.append(dt)
            derivative_list.append(derivative)
        axes.plot([self.capture_dict["max_derivative_time"], self.capture_dict["max_derivative_time"]], [-0.1e-5, 1e-5], linestyle="--", color="grey")
        axes.plot([self.capture_dict["10_pc_time"], self.capture_dict["10_pc_time"]], [-0.1e-5, 1e-5], linestyle="--", color="grey")
        axes.plot([self.capture_dict["first_minimum_time"], self.capture_dict["first_minimum_time"]], [-0.1e-5, 1e-5], linestyle="--", color="grey")
        axes.plot(dt_list, derivative_list)
        axes.plot([dt_list[0], dt_list[-1]], [0,0], linestyle="--")
        axes.set_xlabel("Time [ns]")
        axes.set_ylabel("Derivative of integrated beam monitor signal [AU]")
        axes.set_ylim([-0.1e-5, 1e-5])
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        figure_sum.savefig(os.path.join(self.output_directory, f"{self.file_prefix}monitor_turn_integral_derivative.png"))
        figure_sum.clear()
        matplotlib.pyplot.close(figure_sum)


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
        axes.text(0.01, 0.9, f"Time window: {t0:.1f} {t1:.1f} [ns]",
                  transform=axes.transAxes)
        axes.set_xlabel("Frequency [GHz]")
        axes.text(0.01, 0.95, self.problem_title,
                  transform=axes.transAxes)
        figure.savefig(os.path.join(self.output_directory, f"{self.file_prefix}fft_{t0:.1f}_{t1:.1f}.png"))
        figure.clear()
        matplotlib.pyplot.close(figure)


    def get_rf_time(self):
        if self.rf_data.chi2 > 0.5:
            dt = self.config["programme"]["time_delay"]
            t_rf_start = self.config["programme"]["t_list"][1]+dt
            t_rf_end = self.config["programme"]["t_list"][2]+dt
            t_programme_end = self.rf_data.data[rf_data.time_key][-1]*1e9
        else:
            t_rf_start = self.rf_data.v_model.t_list[1]
            t_rf_end = self.rf_data.v_model.t_list[2]
            t_programme_end = self.rf_data.data[self.rf_data.time_key][-1]*1e9
        print("PROGRAMME END", t_programme_end)
        return t_rf_start, t_rf_end, t_programme_end

    def get_max_value(self, t_start, t_end, t_step):
        t0 = t_start
        value_list = []
        time_list = []
        while t0 < t_end+t_step:
            time_list.append(t0)
            value_list.append(float(self.integral_fit_function(t0)))
            t0 += t_step
        max_value = max(value_list)
        max_index = value_list.index(max_value)
        max_time = time_list[max_index]
        update = {
            "max_time":max_time,
            "max_value":max_value
        }
        return update

    def process_derivative(self, t_start, t_end, t_step):
        t0 = t_start
        derivative_list = []
        time_list = []
        while t0 < t_end+t_step:
            t1 = t0+t_step
            i0 = self.integral_fit_function(t0)
            i1 = self.integral_fit_function(t1)
            derivative = (i1-i0)/(t1-t0)
            derivative_list.append(derivative)
            time_list.append((t1+t0)/2)
            t0 += t_step
        max_derivative = max(derivative_list)
        max_index = derivative_list.index(max_derivative)
        max_time = time_list[max_index]
        update = {
            "max_derivative_time":max_time,
            "max_derivative_value":max_derivative,
        }

        update["first_minimum_time"] = -1
        for i in range(max_index, len(derivative_list)-1):
            if derivative_list[i+1] > derivative_list[i]:
                update["first_minimum_time"] = time_list[i]
                update["first_minimum_rf_voltage"] = self.rf_data.v_model.get_voltage_magnitude(time_list[i]).item()
                break

        update["10_pc_time"] = -1
        for i in range(max_index, len(derivative_list)-1):
            if derivative_list[i] < max_derivative*0.1:
                update["10_pc_time"] = time_list[i]
                update["10_pc_voltage"] = self.rf_data.v_model.get_voltage_magnitude(time_list[i]).item()
                break

        return update

    def process_monitor_data(self):
        t_ramp_start, t_ramp_end, t_programme_end = self.get_rf_time()
        self.capture_dict = self.get_max_value(t_ramp_start, t_programme_end, 1e4)
        update = self.process_derivative(t_ramp_start, t_ramp_end, 1e4)
        self.capture_dict.update(update)
        print(json.dumps(self.capture_dict, indent=2))
