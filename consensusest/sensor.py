import logging
import numpy as np
import numpy.random as rand
import numpy.linalg as LA

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
    """
    Using notation from Olfati-Saber (2007)
    Note: No time dependence is present
    model     - H
    meas_cov  - R
    """
    def __init__(self, model, meas_cov, data=None):
        model = np.matrix(model)
        meas_cov = np.matrix(meas_cov)
        assert(meas_cov.shape[0] == meas_cov.shape[1])
        assert(meas_cov.shape[0] == model.shape[0])
        self.model = model
        self.meas_cov = meas_cov
        self.H = self.model
        self.R = self.meas_cov
        self.R_inv = LA.inv(self.meas_cov)
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

    def compute_locals_alg3(self):
        logging.debug('H\' shape: {}'.format(self.H.T.shape))
        logging.debug('R-1 shape: {}'.format(self.R_inv.shape))
        logging.debug('zi  shape: {}'.format(self.meas.shape))
        self.u = np.dot(np.dot(self.H.T, self.R_inv), self.meas)
        self.U = np.dot(np.dot(self.H.T, self.R_inv), self.H)
        logging.debug('u shape: {}'.format(self.u.shape))
        logging.debug('U shape: {}'.format(self.U.shape))
        logging.debug('U: \n{}'.format(self.U))

    #def prepare_update(self, step, nbr_meas, nbr_cov, nbr_est):
    #    raise NotImplementedError("Implementation paused")
    #    self.y = sum(agg_meas, axis=0) + self.meas[step]
    #    #self.S = agg_


	
                
    def __str__(self):
        s = "Model: {}\n".format(self.model)
        if not self.meas is None:
            return s + "Values: {}".format(self.meas)
        else:
            return s




