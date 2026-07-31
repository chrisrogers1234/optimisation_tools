"""
Returns score given changes from on momentum orbit
"""
import optimisation_tools.analysis.optimiser.score

class OffMomentumScore(optimisation_tools.analysis.optimiser.score.Score):
    def __init__(self, beam_x_key, beam_y_key, delta_x_key, delta_y_key):
        self.keys = [beam_x_key, beam_y_key, delta_x_key, delta_y_key]
        self.beam_x_p_plus = 0.0
        self.beam_y_p_plus = 0.0

    def get_score(self, hit_list_of_lists):
        score = 0.0
        x_target = self.beam_x + self.beam_x_p_plus
        y_target = self.beam_y + self.beam_y_p_plus
        for station in [1, 3]:
            for hit in hit_list_of_lists[0]:
                if hit["station"] != station:
                    continue
                score += (hit["x"] - x_target)**2
                score += (hit["y"] - y_target)**2
                print(f"Orbit station: {hit['station']} x: {hit['x']:8.4g}  px:  {hit['px']:8.4g} y: {hit['y']:8.4g} py:  {hit['py']:8.4g} ** target x: {x_target:8.4g} y: {y_target:8.4g}")
        return score

    def receive_output(self, post_dict):
        self.beam_x = post_dict[self.keys[0]] # on momentum orbit
        self.beam_y = post_dict[self.keys[1]] # on momentum orbit
        self.beam_x_p_plus = post_dict[self.keys[2]] # delta to off-momentum orbit
        self.beam_y_p_plus = post_dict[self.keys[3]] # delta to off-momentum orbit

    def get_input_psv_list(self):
        return [{"x":self.beam_x_p_plus, "y":self.beam_y_p_plus}] # energy is set by tracking

    def save_score(self):
        raise NotImplementedError("Should be overridden")

    def post_output(self): # optionally, we post output from the score function
        return {}

