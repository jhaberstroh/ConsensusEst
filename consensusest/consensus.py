import logging
import numpy as np
import numpy.linalg as LA
from copy import deepcopy

def kalmansmoother_twosensor(data, sensors, duration, dyn_model, 
        dyn_noise_model, dyn_noise_cov, iters, ep = .1):
    """
    """
    for s in sensors:
        s.add_data(data)
        s.compute_locals_alg3()

    A = dyn_model
    B = dyn_noise_model
    Q = dyn_noise_cov
    
    x_est = []
    P_est = []
    for s in sensors:
        model_dims = s.model.shape[1]
        meas_dims  = s.model.shape[0]
        logging.debug("Model dims: {}".format(model_dims))
        logging.debug("Meas dims: {}".format(meas_dims))
        x_est.append(np.zeros((model_dims,1)))
        P_est.append(np.eye(model_dims) * 1000.)

    graph = {0: [1],
             1: [0]}

    for t in xrange(duration):
        print "t = {}: {}".format(t, data[:,t])
        x_prev = deepcopy(x_est)
        P_prev = deepcopy(P_est)
        for i in graph.keys():
            yi = sensors[i].u[:,t]
            Si = sensors[i].U[:,:]
            for nbr in graph[i]:
                yi += sensors[nbr].u[:,t]
                Si += sensors[nbr].U[:,:]
            for it in xrange(iters):
                Mi = LA.inv(LA.inv(P_prev[i]) + Si)
                x_hat = x_prev[i] + np.dot(Mi, (yi - np.dot(Si, x_prev[i]) ) )
                for nbr in graph[i]:
                    x_hat += np.dot(Mi, (x_prev[nbr] - x_prev[i])) * ep
                # Update the state of the filyter
                P_est[i] = np.dot(A, np.dot(Mi, A.T)) + np.dot(B, np.dot(Q, B.T))
                x_est[i] = np.dot(A, x_hat)
            print "Current estimate from {}: {}".format(i, x_est[i].T)
      

   
