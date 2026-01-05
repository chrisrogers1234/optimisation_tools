import numpy

import optimisation_tools.toy_model.longitudinal_model.beam as beam
import beam.particle

class BeamFactory(object):
    def __init__(self):
        pass

    @classmethod
    def make_coasting_beam_square(cls, n_events, low_energy, high_energy, n_turns, model):
        tof_list = []
        e_list = numpy.random.uniform(low_energy, high_energy, n_events)
        p_list = [beam.particle.Particle(0, e, cls.mass) for e in e_list]
        for p in p_list:
            tof = model.get_time_of_flight(p)
            p.t = numpy.random.uniform(0.0, tof*n_turns)
            tof_list.append(tof)
        print("f:", 1./min(tof_list), 1./max(tof_list))
        return p_list

    @classmethod
    def make_uniform_distribution(cls, n_events, low_time, high_time, low_energy, high_energy):
        p_list= []
        for i in range(n_events):
            p_list.append(cls.make_p(numpy.random.uniform(low_time, high_time), numpy.random.uniform(low_energy, high_energy)))
        return p_list

    @classmethod
    def make_p(cls, t, energy):
        p = Particle(t, energy, BeamFactory.mass)
        return p

    mass = 938.27208943
