
class Synapses():
    """
    Synapses class.
    Contains their 'microscopic' properties.
    """

    def __init__(self, type) -> None:

        # Cell type
        self.type = type

        # Synaptic properties
        self.params = self._getParams(type)

    @staticmethod
    def _getParams(type):
        """
        Get synapse parameters from cell type.
        """

        # Inhibitory cell
        if type == 'FS':
            params = {
                'Q': 5*1e-9,
                'T': 5*1e-3,
                'E': -80*1e-3
            }

        # Excitatory cell
        elif type == 'RS':
            params = {
                'Q': 1*1e-9,
                'T': 5*1e-3,
                'E': 0
            }

        else:
            raise RuntimeError("Unknown neuron type. Expected 'FS' or 'RS'")

        return params

    def getParams(self):
        """
        Return synaptic properties.
        """
        return self.params
