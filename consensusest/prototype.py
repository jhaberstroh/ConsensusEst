import consensusest.gendata as gen
import consensusest.sensor as sense
import consensusest.consensus as con
import numpy as np
import logging

def main():
    logging.basicConfig(level=logging.DEBUG)
    dim = 2
    r = gen.gausswalker(100, dims=dim)

    xm = sense.Sensor([[1, 0]], [[.1]])
    ym = sense.Sensor([[0, 1]], [[.1]])

    dyn_model = np.eye(2)
    dyn_noise_model = np.eye(2)
    dyn_noise_cov = np.eye(2)

    con.kalmansmoother_twosensor(r.T, sensors=(xm,ym), duration=100, 
            dyn_model=dyn_model, dyn_noise_model=dyn_noise_model,
            dyn_noise_cov=dyn_noise_cov, iters = 10)

if __name__=="__main__":
    main()
