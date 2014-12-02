import logging
import numpy as np
import numpy.linalg as LA
from copy import deepcopy

def kalmansmoother_network(data, sensors, duration, dyn_model, 
        dyn_noise_model, dyn_noise_cov, iters, ep = 1, graph=None):
    """
    """
    # If no graph is supplied, full connectivity will be given
    if graph is None:
        graph = dict()
        n_sensor = len(sensors)
        for node in xrange(n_sensor):
            graph[node] = range(0,node) + range(node+1, n_sensor)

    # Add the same data to the sensors and pre-compute the locals
    for s in sensors:
        s.add_data(data)
        s.compute_locals_alg3()

    A = dyn_model
    B = dyn_noise_model
    Q = dyn_noise_cov
    
    # Create the node-to-node estimates that will be updated
    x_est = []
    P_est = []
    for s in sensors:
        model_dims = s.model.shape[1]
        meas_dims  = s.model.shape[0]
        logging.debug("Model dims: {}".format(model_dims))
        logging.debug("Meas dims: {}".format(meas_dims))
        x_est.append(np.zeros((model_dims,1)))
        P_est.append(np.eye(model_dims) * 1000.)

    # Run the filter through the measurements
    for t in xrange(duration):
        print "t = {}: {}".format(t, data[:,t])
        x_prev = deepcopy(x_est)
        P_prev = deepcopy(P_est)
        for i in graph.keys():
            yi = sensors[i].u[:,t]
            Si = sensors[i].U[:,:]
            for nbr in graph[i]:
                # Do NOT use +=, it will modify the objects
                # within the elements
                yi = yi + sensors[nbr].u[:,t]
                Si = Si + sensors[nbr].U[:,:]
            for it in xrange(iters):
                Mi = LA.inv(LA.inv(P_prev[i]) + Si)
                x_hat = x_est[i] + np.dot(Mi, (yi - np.dot(Si, x_est[i]) ) )
                for nbr in graph[i]:
                    x_hat += np.dot(Mi, (x_est[nbr] - x_est[i])) * ep
                # Update the state of the filter
                P_est[i] = np.dot(A, np.dot(Mi, A.T)) + np.dot(B, np.dot(Q, B.T))
                x_est[i] = np.dot(A, x_hat)
            print "Current estimate from {}: {}".format(i, x_est[i].T)
            #print "Current error {}: \n{}".format(i, P_est[i])



