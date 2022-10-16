
class Network():
    """
    Network class.

    Parameters:
        - X_discretization, Z_discretization: in mm.
        - X_extent, Z_extent: in mm.
        - exc_connect_extent, inh_connect_extent: in mm.
        - conduction_velocity: in mm/s
    """

    def __init__(self, network, custom_params=None) -> None:

        # Default parameters
        self.type = network
        self.params = {
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

        # Eventually override default params
        if custom_params:
            for key, val in custom_params.items():
                self.params[key] = val

        # Discretize model parameters
        self._mm2pixels()

    def _mm2pixels(self):
        """
        Convert all network parameters from mm to pixels.
        """

        ratio = self.params['X_extent']/self.params['X_discretization']

        # Conversion
        self.params['exc_decay_connect'] = self.params['exc_connect_extent']/ratio
        self.params['inh_decay_connect'] = self.params['inh_connect_extent']/ratio

        # In practice connectivity extends up to 3 std dev.
        self.params['exc_connected_neighbors'] = int(
            3*self.params['exc_decay_connect']/ratio
        )
        self.params['inh_connected_neighbors'] = int(
            3*self.params['inh_decay_connect']/ratio
        )
        self.params['conduction_velocity'] = self.params['conduction_velocity']/ratio

        return

    def getParams(self):
        """
        Return network parameters.
        """
        return self.params
