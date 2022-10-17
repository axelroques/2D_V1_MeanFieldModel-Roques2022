
class Network():
    """
    Network class.

    Parameters:
        - X_discretization, Z_discretization: in mm.
        - X_extent, Z_extent: in mm.
        - exc_connect_extent, inh_connect_extent: in mm.
        - conduction_velocity: in mm/s
    """

    def __init__(self, network, custom_parameters=None) -> None:

        # Default parameters
        self.type = network
        self.parameters = {
            'Ntot': 10000,
            'gei': 0.2,
            'X_discretization': 30.,
            'X_extent': 36.,
            'Z_discretization': 30.,
            'Z_extent': 36.,
            'exc_connect_extent': 5.,
            'inh_connect_extent': 1.,
            'conduction_velocity': 300.
        }

        # Eventually override default parameters
        if custom_parameters:
            for key, val in custom_parameters.items():
                self.parameters[key] = val

        # Discretize model parameters
        self._mm2pixels()

    def _mm2pixels(self):
        """
        Convert all network parameters from mm to pixels.
        """

        ratio = self.parameters['X_extent']/self.parameters['X_discretization']

        # Conversion
        self.parameters['exc_decay_connect'] = self.parameters['exc_connect_extent']/ratio
        self.parameters['inh_decay_connect'] = self.parameters['inh_connect_extent']/ratio

        # In practice connectivity extends up to 3 std dev.
        self.parameters['exc_connected_neighbors'] = int(
            3*self.parameters['exc_decay_connect']/ratio
        )
        self.parameters['inh_connected_neighbors'] = int(
            3*self.parameters['inh_decay_connect']/ratio
        )
        self.parameters['conduction_velocity'] = self.parameters['conduction_velocity']/ratio

        return

    def getParameters(self):
        """
        Return network parameters.
        """
        return self.parameters
