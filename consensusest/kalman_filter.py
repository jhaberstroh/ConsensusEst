import numpy as np
import scipy as sp
import numpy.random


class kalman_filter:         # must hold state as calculation is streaming
    def __init__(self, A,G,Q,C,R):
        # x_{t+1} = A x_{t} + G w_{t}
        # w_{t} = C x_{t} + v_{t}
        
        self.A = A
        self.G = G
        self.Q = Q
        self.C = C
        self.R = R

        self.Xest = np.zeros((np.shape(A)[1],1))

        self.P = np.identity(np.shape(A)[1])

    def update(self, new_obs):
        self.Xest = self.A.dot(self.Xest)

        # P =  A . P . At  +  G . Q . Gt
        self.P = (self.A.dot(self.P)).dot(np.transpose(self.A)) + \
                (self.G.dot(self.Q)).dot(np.transpose(self.G))
        
        # K = P . Ct . (C . P . Ct)-1
        self.K = self.P.dot(np.transpose(self.C)).dot(np.linalg.inv
                (self.C.dot(self.P).dot(np.transpose(self.C)) + self.R))

        self.Xest = self.Xest + self.K.dot(new_obs - self.C.dot(self.Xest))
        
        self.P = self.P - self.K.dot(self.C).dot(self.P)

    def update_noisy(self, new_obs):
        new_obs = np.array(new_obs)
        zero_mean = np.zeros(self.R.shape[0])
        noise = np.random.multivariate_normal(zero_mean, self.R)
        new_obs += noise
        new_obs.shape = (len(new_obs), 1)
        self.update(new_obs)
