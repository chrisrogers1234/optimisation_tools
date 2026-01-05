import json

class PostprocessingOutput:
    def __init__(self):
        self.output = {}
        self.output_filename = None
        self.rf_data = None
        self.my_analysis = None
        self.config = None

    def process(self):
        self.process_rf_data()
        self.process_monitor_data()
        self.write()

    def process_rf_data(self):
        rf_data = {
            "t_list":self.rf_data.v_model.t_list,
            "v_list":self.rf_data.v_model.v_list,
            "chi2":self.rf_data.chi2,
        }
        self.output["rf_data"] = rf_data

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
            value_list.append(float(self.my_analysis.integral_fit_function(t0)))
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
            i0 = self.my_analysis.integral_fit_function(t0)
            i1 = self.my_analysis.integral_fit_function(t1)
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
        capture_dict = self.get_max_value(t_ramp_start, t_programme_end, 1e4)
        update = self.process_derivative(t_ramp_start, t_ramp_end, 1e4)
        capture_dict.update(update)
        self.output["beam_capture"] = capture_dict
        print(self.output["beam_capture"])

    def write(self):
        with open(self.output_filename, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.output, indent=2))

