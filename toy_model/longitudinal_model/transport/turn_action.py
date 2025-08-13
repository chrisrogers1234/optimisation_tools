import matplotlib.pyplot
import optimisation_tools.toy_model.longitudinal_model.analysis as analysis
import analysis.plot_beam

class TurnAction(object):
    def __init__(self, rf_program, monitor, model, plot_contours):
        self.program = rf_program
        self.monitor = monitor
        self.model = model
        self.plot_contours = plot_contours
        self.plot_frequency = 100
        self.output_directory = "output/test"
        self.plotter = analysis.plot_beam.PlotBeam(self.output_directory, model)

    def do_turn_action(self, turn, particle_collection):
        suffix = str(turn).rjust(4, "0")
        self.plotter.output_directory = self.output_directory
        if turn % self.plot_frequency == 0:
            print("Doing turn action on turn", turn)
            figure = self.plotter.plot_beam(particle_collection, self.plot_contours, suffix)
            matplotlib.pyplot.close(figure)
        self.monitor.do_one_turn(turn, particle_collection)
