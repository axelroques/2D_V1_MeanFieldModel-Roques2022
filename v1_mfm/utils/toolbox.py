
import numpy as np


def gaussian_connectivity(x, x0, dx, normalization):
    """
    Determine the weight of the connectivity between each neighbouring network.
    """
    return normalization / (
        np.sqrt(2.*np.pi) * (dx+1e-12)) *\
        np.exp(-(x-x0)**2 /2/(1e-12+dx)**2
               )
