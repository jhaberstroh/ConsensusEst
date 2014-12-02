import numpy as np
import numpy.random as rand

def gausswalker(times, dims):
    """
    Create a timesieres of a gaussian random walk in n dimensions.
    Returns a numpy array of shape (times, dims).
    """
    rand_vals = rand.randn(times, dims)
    return np.cumsum(rand_vals, axis=0)
