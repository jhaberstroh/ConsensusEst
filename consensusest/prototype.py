import consensusest.gendata as gen
import consensusest.sensor as sense
import consensusest.kalman_filter as kalman
import numpy as np
import numpy.random
import numpy.linalg as LA
import logging
import argparse 

def main():
    parser = argparse.ArgumentParser(
            description='Simulation for orthogonal network filters, ' + 
            'comparing filters to a kalman filter of the same data rate.')
    parser.add_argument('-d', type=int, default=2,
            help='Number of dimensions to simulate')
    parser.add_argument('-t', type=int, default=200, 
            help='Amount of time to simulate')
    args = parser.parse_args()
    #logging.basicConfig(level=logging.DEBUG)
    time = args.t
    dim = args.d
    r = gen.gausswalker(time, dims=dim)

    dyn_model = np.eye(dim)
    dyn_noise_model = np.eye(dim)
    dyn_noise_cov = np.eye(dim)
    noise_amp = .1

    x0 = np.array([[0]*dim]).T
    P0 = np.eye(dim) * 1000.
    net = sense.LiveSensorNetwork(x0, P0)
    for i in xrange(dim):
        meas = np.array([np.eye(dim)[i,:]])
        sensor = sense.Sensor(meas, [[noise_amp]])
        net.add_sensor(sensor, range(i))

    net_rand = sense.LiveSensorNetwork(x0, P0)
    for i in xrange(dim):
        meas = np.random.random((dim,dim))
        sensor = sense.Sensor(meas, np.eye(dim) * noise_amp)
        net_rand.add_sensor(sensor, range(i))
    
    meas_model = np.eye(dim)
    meas_cov = np.eye(dim) * noise_amp

    kf = kalman.kalman_filter(A = dyn_model, G = dyn_noise_model,
            Q = dyn_noise_cov, C = meas_model, R = meas_cov)
    
    rms_KF_tot = 0 
    rms_net_tot = 0
    rms_rand_tot = 0

    for t in xrange(time):
        #print 'time {}: {}'.format(t, r[t,:])
        net.stream_data(r[t,:])
        net.iterate_filter(dyn_model, dyn_noise_model, dyn_noise_cov)
        net_rand.stream_data(r[t,:])
        net_rand.iterate_filter(dyn_model, dyn_noise_model, dyn_noise_cov)
        kf.update_noisy(r[t,:])
        
        #print "Estimates: "
        #print net[0].T, net[1].T
        #print kf.Xest.T

        rms_KF_tot  += LA.norm(kf.Xest.T - r[t,:])
        for i in xrange(dim):
            rms_net_tot += LA.norm(net[i].T - r[t,:]) / dim
        for i in xrange(dim):
            rms_rand_tot += LA.norm(net_rand[i].T - r[t,:]) / dim

    print rms_KF_tot / dim / time, rms_net_tot / dim / time, rms_rand_tot / dim / time

if __name__=="__main__":
    main()
