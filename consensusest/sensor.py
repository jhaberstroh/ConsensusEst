import numpy as np
import numpy.random as rand

class Measurement:
    def __init__(self, dir, data):
        self.dir = dir
        self.data = data

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
    direction = np.zeros(data.shape[1], dtype=np.float)
    direction[dir] = 1.
    return Measurement( direction,
            true_value + ( rand.randn(data.shape[0]) * noise_amp ))



class Sensor:
    def __init__(self, model, meas_cov, data=None):
        model = np.matrix(model)
        meas_cov = np.matrix(meas_cov)
        assert(meas_cov.shape[0] == meas_cov.shape[1])
        assert(meas_cov.shape[0] == model.shape[0])
        self.model = model
        self.meas_cov = meas_cov
        if not data is None:
            self.add_data(data)
        else:
            self.meas = None
    def add_data(self, data, cols_are_time=True):
        data = np.array(data)
        if not cols_are_time:
            data = data.T
        assert(data.shape[0] == self.model.shape[1])
        n_randoms = data.shape[1]
        zero_mean = np.zeros(self.model.shape[0])
        noise = rand.multivariate_normal(zero_mean, self.meas_cov, n_randoms)
        # Store separate samples as different columns
        noise = noise.T

        self.meas = np.dot(self.model, data) 
        self.meas += noise
                
    def __str__(self):
        s = "Model: {}\n".format(self.model)
        if not self.meas is None:
            return s + "Values: {}".format(self.meas)
        else:
            return s


