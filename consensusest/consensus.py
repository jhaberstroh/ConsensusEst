import numpy as np

def cartesianfilter(dims, measurements):
    """
    Filter to use consensus estimation on cartesian data to infer the
    position of an object in n dimensions.

    dims: int, number of dimensions
    measurements: array of directions that observations were made of

    Returns the filtered signal.
    """
    size = None
    for a in measurements:
        if size is None:
            size = a.data.shape[0]
        if (a.data.shape[0] != size):
            raise ValueError("Measurements to cartesian consensus filter" +
                    " must be the same length")
        if (len(a.dir) != dims):
            raise ValueError("Measurements to cartesian consensus filter" +
                    " must be of dimension dim = {}, found {}"
                    .format(dims, len(a.dir)))
    raise NotImplementedError("Cartesian filter not implemented")

def distkalman(sensors):
    raise NotImplementedError("Distributed Kalman not implemented")
