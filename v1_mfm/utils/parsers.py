
import numpy as np


def parseNetworkParameters(network_parameters):
    """
    Simple helper function that iterates over network 
    parameters and returns relevant parameters for other 
    functions.
    """

    # Model
    exc_connected_neighbors = network_parameters['exc_connected_neighbors']
    inh_connected_neighbors = network_parameters['inh_connected_neighbors']
    exc_decay_connect = network_parameters['exc_decay_connect']
    inh_decay_connect = network_parameters['inh_decay_connect']

    # Construction of neighbour's 2D matrices
    # Excitatory population
    radius_exc = exc_connected_neighbors
    y, x = np.ogrid[-radius_exc: radius_exc+1, -radius_exc: radius_exc+1]
    Xn_exc = np.sqrt(x**2+y**2)
    for row in range(len(Xn_exc[0])):
        for col in range(len(Xn_exc[1])):
            if Xn_exc[row, col] > radius_exc:
                Xn_exc[row, col] = -99
            else:
                if col <= len(Xn_exc[1])//2-1:
                    Xn_exc[row, col] = - Xn_exc[row, col]  # - sign on the left
                else:
                    Xn_exc[row, col] = Xn_exc[row, col]  # + sign on the right
    # Inhibitory population
    radius_inh = inh_connected_neighbors
    y, x = np.ogrid[-radius_inh: radius_inh+1, -radius_inh: radius_inh+1]
    Xn_inh = np.sqrt(x**2+y**2)
    for row in range(len(Xn_inh[0])):
        for col in range(len(Xn_inh[1])):
            if Xn_inh[row, col] > radius_inh:
                Xn_inh[row, col] = -99
            else:
                if col <= len(Xn_inh[1])//2-1:
                    Xn_inh[row, col] = - Xn_inh[row, col]
                else:
                    Xn_inh[row, col] = Xn_inh[row, col]

    # Construction of X and Z vectors
    X, Z, = getXZVectors(network_parameters)

    return X, Z, Xn_exc, Xn_inh, \
        exc_decay_connect, inh_decay_connect


def getXZVectors(network_parameters):
    """
    Return vectors X and Z.
    """

    X = np.linspace(
        0,
        network_parameters['X_extent'],
        int(network_parameters['X_discretization']),
        endpoint=True
    )
    Z = np.linspace(
        0,
        network_parameters['Z_extent'],
        int(network_parameters['Z_discretization']),
        endpoint=True
    )

    return X, Z
