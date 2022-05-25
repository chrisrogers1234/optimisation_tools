import toy_model
import toy_model.toy_model
import matplotlib

def setup_model():
    toy_model.toy_model.ToyModel.amp_scale = 0.25
    model = toy_model.toy_model.ToyModel()
    model.pulse_emittance = 0.0
    model.dp_model = "none" # none gauss or fixed
    model.foil_column_density = 0 #20e-6 # g/cm^3
    model.closed_orbit = "scripts/toy_model/tests/dispersion/closed_orbits_cache"
    model.output_dir = "output/toy_model_tests/transport/"
    model.settings = ""
    model.number_pulses = 2
    model.default_bumps = [[1.0, 0.025, 1.0, 0.025], [1.0, 0.15, 1.0, 0.15]]
    model.max_turn = 50
    model.number_per_pulse = 1
    model.turn_bumper_index = [range(model.number_pulses), range(model.number_pulses)]
    model.plot_frequency = 1
    model.accumulate = True
    model.setup_material()
    return model

def dispersion_test():
    model = setup_model()
    model.initialise()
    model.run_toy_model()
    model.finalise()


if __name__ == "__main__":
    dispersion_test()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
