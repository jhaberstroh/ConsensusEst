import consensusest.gendata as gen
import consensusest.sensor as sense

def main():
    y = gen.gausswalker(100,2)
    print y
    ym = sense.cartesiansensor(y, 0, .1)
    print ym



if __name__ == "__main__":
    main()
