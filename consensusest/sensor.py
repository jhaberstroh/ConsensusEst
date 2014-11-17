import numpy as np
import numpy.random as rand

def cartesiansensor(data, dir, noise_amp = .5):
    """
    Takes in numpy array of shape (times, dims) and returns a noisy measure
    of data[:, dir].
    """
    if dir >= data.shape[1]:
        raise ValueError(
                "In cartesiansensor, requested dimension {} of {} size array."
                .format(data.shape[1], dir))

    true_value = data[:,dir]
    return true_value + ( rand.randn(data.shape[0]) * noise_amp )
