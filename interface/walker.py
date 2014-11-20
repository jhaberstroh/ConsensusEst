import consensusest.gendata as gen
import consensusest.sensor as sense
import consensusest.consensus as con
import numpy as np

def main():
    dim = 2
    r = gen.gausswalker(100, dims=dim)

    xm = sense.Sensor([[1, 0]], [[.1]], r.T)
    ym = sense.Sensor([[0, 1]], [[.1]], r.T)

    print xm
    print ym

    #ym = sense.cartesiansensor(r, dir=0, noise_amp=.1)
    #xm = sense.cartesiansensor(r, dir=1, noise_amp=.1)
    #print ym.data
    #print ym.dir
    #print xm.dir
    #con.cartesianfilter(dims=dim, measurements=(xm, ym))

    con.distkalman(sensors=(xm,ym))


    




if __name__ == "__main__":
    main()
