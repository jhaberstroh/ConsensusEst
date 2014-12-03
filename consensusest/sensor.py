import logging
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from copy import deepcopy

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
        if data.shape[0] != self.model.shape[1]:
            raise ValueError("Data cannot be loaded into sensor, dimensions "+
                    "do not match: \ndata axis 0: {}\n".format(data.shape[0]) +
                    "model axis 1: {}".format(self.model.shape[1]))
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



class LiveSensorNetwork:
    def __init__(self, init_pos, init_err):
        """
        Initialize sensor network with the initial position and error
        that sensors without data should believe in. Namely, the init_err
        should be large and the init_pos should be sensible and finite.
        """
        init_pos = np.array(init_pos)
        init_err = np.array(init_err)
        assert(init_pos.shape[0] == init_err.shape[0])
        self.sensors = []
        self.network = dict()
        self.x_est = []
        self.P_est = []
        self.x0 = init_pos
        self.P0 = init_err
    def add_sensor(self, sensor, connections = [], x0 = None, P0 = None):
        """
        Takes a sensor object and an array of connections (specified 
        numerically). Updates all other sensors so that the network is 
        an undirected graph.

        Sensor network uses 0-based indexing.

        Leave x0 and P0 as none if you want to initialize with default
        values for initial position and uncertainty.
        """
        # TODO: Check that the measurement model has a valid shape, relative
        # to the x0 and P0
        # Update sensors and sensor estimates
        self.sensors.append(sensor)
        self.x_est.append(self.x0) if x0 is None else self.x_est.append(x0)
        self.P_est.append(self.P0) if P0 is None else self.P_est.append(P0)
        # Update connections
        N = len(self.sensors) - 1
        self.network[N] = connections
        for n in connections:
            self.network[n].append(N)
    def update_network(self, new_network):
        """
        Reinitialize the network with a specified symmetric configuration.
        """
        # Verify symmetry in the connections
        for k in new_network.keys():
            for j in new_network[k]:
                raise ValueError("In LiveSensorNetowrk.update_network: " + 
                        "Network must be symmetric")
        self.network = new_network
    def stream_data(self, data, sensor_list=None):
        """
        Loads data in each sensor of the sensor_list. All previous data
        is overwritten.
        """
        if sensor_list is None:
            sensor_list = range(len(self.sensors))

        data = np.array(data)
        data.shape = (len(data), 1)
        print data.shape[1]

        for s in sensor_list:
            self.sensors[s].add_data(data)
            self.sensors[s].compute_locals_alg3()
    def iterate_filter(self, dyn_model, dyn_noise_model, dyn_noise_cov,
            iters = 1, epsilon = 1):
        A = dyn_model
        B = dyn_noise_model
        Q = dyn_noise_cov
        x_prev = deepcopy(self.x_est)
        P_prev = deepcopy(self.P_est)
        for i in self.network.keys():
            yi = self.sensors[i].u[:,0]
            Si = self.sensors[i].U[:,:]
            for nbr in self.network[i]:
                # Do NOT use +=, it will modify the objects
                # within the elements
                yi = yi + self.sensors[nbr].u[:,0]
                Si = Si + self.sensors[nbr].U[:,:]
                logging.debug(self.sensors[nbr].U[:,:])
            for it in xrange(iters):
                Mi = LA.inv(LA.inv(P_prev[i]) + Si)
                x_hat = self.x_est[i] + \
                        np.dot(Mi, (yi - np.dot(Si, self.x_est[i]) ) )
                for nbr in self.network[i]:
                    x_hat += np.dot(Mi, (self.x_est[nbr] - self.x_est[i])) * epsilon
                # Update the state of the filter
                self.P_est[i] = np.dot(A, np.dot(Mi, A.T)) + np.dot(B, np.dot(Q, B.T))
                self.x_est[i] = np.dot(A, x_hat)
            print "Current estimate from {}: {}".format(i, self.x_est[i].T)
    def __getitem__(self, index):
        return self.x_est[index]



