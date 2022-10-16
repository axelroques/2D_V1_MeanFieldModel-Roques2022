
from ..utils.stimulation_types import stim_MOV_HORIZ_LINE
from ..utils.stimulation_types import stim_MOV_VERT_LINE
from ..utils.stimulation_types import stim_LINE_MOTION
from ..utils.stimulation_types import stim_DOUBLE_LINE
from ..utils.stimulation_types import stim_VERT_LINE
from ..utils.stimulation_types import stim_RECTANGLE
from ..utils.stimulation_types import stim_DOT_RIGHT
from ..utils.stimulation_types import stim_DOT_LEFT
from ..utils.stimulation_types import stim_CENTER
from ..utils.stimulation_types import stim_CIRCLE
from ..utils.stimulation_types import stim_DOUBLE
from ..utils.stimulation_types import stim_RANDOM
from ..utils.stimulation_types import stim_SMILEY
from ..utils.stimulation_types import stim_DOT


import numpy as np


class Stimuli():
    """
    Stimuli class.
    Contains stimulation parameters.
    """

    def __init__(
        self,
        t, X, Z,
        stimulation_type='CENTER',
        custom_params={}
    ) -> None:

        # Default stimulation parameters
        self.params = {
            't': t,
            'stimulation_type': stimulation_type,
            'sX': 1.5,  # extension of the stimulus in X
            'sZ': 1.5,  # extension of the stimulus in Z
            'tstop': 400e-3,
            'tstart': 150e-3,
            'amp': 15.,
            'Tau1': 50e-3,
            'Tau2': 150e-3
        }

        # Eventually override default parameters
        for key, val in custom_params.items():
            self.params[key] = val

        # Time vector for the stimulations

        self.params.update({'t': t})

        # Generate stimulation
        self._generateStimuli(X, Z)

    def _generateStimuli(self, X, Z):
        """
        Returns the 3D array containing the afferent 
        stimulation for the model.
        """

        stimulation = {
            'CENTER': stim_CENTER,
            'DOT': stim_DOT,
            'DOT_LEFT': stim_DOT_LEFT,
            'DOT_RIGHT': stim_DOT_RIGHT,
            'DOUBLE': stim_DOUBLE,
            'MOV_HORIZ_LINE': stim_MOV_HORIZ_LINE,
            'MOV_VERT_LINE': stim_MOV_VERT_LINE,
            'VERT_LINE': stim_VERT_LINE,
            'LINE_MOTION': stim_LINE_MOTION,
            'DOUBLE_LINE': stim_DOUBLE_LINE,
            'SMILEY': stim_SMILEY,
            'RECTANGLE': stim_RECTANGLE,
            'CIRCLE': stim_CIRCLE,
            'RANDOM': stim_RANDOM
        }

        try:
            self.Fe_aff = stimulation[self.params['stimulation_type']](
                self.params['t'], X, Z, self.params,
                self._triple_gaussian
            )
        except KeyError:
            raise RuntimeError('Unknown stimulation type.')

        return

    @staticmethod
    def _heaviside(x):
        """
        Simple Heaviside function
        """
        return 0.5*(1+np.sign(x))

    @staticmethod
    def _triple_gaussian(t, X, Z, t0, T1, T2, X0, Z0, sX, sZ, amplitude):
        """
        Equation of the function used to make the afferent input: Gaussian profile in space and time
        """
        return amplitude*(
            np.exp(-(t-t0)**2/2./T1**2)*Stimuli._heaviside(-(t-t0)) +
            np.exp(-(t-t0)**2/2./T2**2)*Stimuli._heaviside(t-t0)) *\
            np.exp(-(X-X0)**2/2./sX**2) *\
            np.exp(-(Z-Z0)**2/2./sZ**2)

    def getParams(self):
        """
        Return stimulation parameters.
        """
        return self.params

    def getAffStim(self):
        """
        Return afferent stimulation.
        """
        return self.Fe_aff
