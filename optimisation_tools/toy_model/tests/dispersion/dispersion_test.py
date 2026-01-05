import toy_model
import toy_model.toy_model
import matplotlib

def setup_model():
    toy_model.toy_model.ToyModel.amp_scale = 0.1
    model = toy_model.toy_model.ToyModel()
    model.pulse_emittance = 0 #0.026
    model.dp_model = "fixed" # none gauss or fixed
    model.dp_over_p = 5e-3 # 0.0013 # 0.0029 # sigma
    model.max_dp_over_p = 3*model.dp_over_p
    model.foil_material = "carbon"
    model.foil_column_density = 0 #20e-6 # g/cm^3
    model.do_scattering = False
    model.do_energy_loss = False
    model.closed_orbit = "scripts/toy_model/tests/dispersion/closed_orbits_cache"
    model.output_dir = "output/toy_model_tests/dispersion/"
    model.settings = ""
    model.number_pulses = 1
    model.default_bumps = [[1.0, 0.0, 1.0, 0.0]]
    model.max_turn = 1
    model.number_per_pulse = 10
    model.turn_bumper_index = [range(1), range(1)]
    model.plot_frequency = 1
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
