
from .synapses import Synapses

import numpy as np
import os


class NeuronPopulation():
    """
    Neuron population class.
    Contains their 'microscopic' properties.
    """

    def __init__(self, type) -> None:

        # Cell type
        self.type = type

        # Get synapse properties
        self.synapses = Synapses(type=self.type)
        self.synapses_parameters = self.synapses.getParameters()

        # Update synapses parameters with population properties
        self.parameters = self.synapses_parameters.copy()
        self._getParameters(type, self.parameters)

    @staticmethod
    def _getParameters(type, synapes_parameters):
        """
        Infer neuron properties from the input cell type.
        """

        absolute_path = os.path.realpath(os.path.dirname(__file__))
        relative_file_path = f'../data/{type}_fit.npy'

        # Inhibitory cell
        if type == 'FS':
            parameters = {
                'Gl': 10.*1e-9,
                'Cm': 200.*1e-12,
                'Trefrac': 5*1e-3,
                'El': -65.*1e-3,
                'Vthre': -50.*1e-3,
                'Vreset': -65.*1e-3,
                'delta_v': 0.5*1e-3,
                'a': 0.,
                'b': 0.,
                'tauw': 1e9*1e-3,
                'p_conn': 0.05/4,
                'P': np.load(os.path.join(absolute_path, relative_file_path))
            }

        # Excitatory cell
        elif type == 'RS':
            parameters = {
                'Gl': 10.*1e-9,
                'Cm': 200.*1e-12,
                'Trefrac': 5*1e-3,
                'El': -65.*1e-3,
                'Vthre': -50.*1e-3,
                'Vreset': -65.*1e-3,
                'delta_v': 2.*1e-3,
                'a': 4.*1e-9,
                'b': 20.*1e-12,
                'tauw': 500.*1e-3,
                'p_conn': 0.05/4,
                'P': np.load(os.path.join(absolute_path, relative_file_path))
            }

        else:
            raise RuntimeError("Unknown neuron type. Expected 'FS' or 'RS'")

        synapes_parameters.update(parameters)

        return

    def getParameters(self):
        """
        Return neuron properties.
        """
        return self.parameters
