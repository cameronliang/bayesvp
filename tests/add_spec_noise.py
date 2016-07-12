import numpy as np
import sys
import pylab as pl


def write_spec(input_path,snr):

    fname = input_path + 'uv_qso.spec'
    wave,flux = np.loadtxt(fname,unpack=True,usecols=[0,1])
    mean_sigma = 1/float(snr)

    new_flux = np.zeros(len(flux))
    error = np.zeros(len(flux))
    for i in range(len(new_flux)):
        error[i] = np.random.normal(mean_sigma,mean_sigma*0.05,1)
        new_flux[i] = flux[i] + np.random.normal(0,mean_sigma,1)

    np.savetxt(input_path + 'uv_qso_' + str(snr) + '.spec',np.c_[wave,new_flux,error])


def main():
    input_path = '/data/jwliang/projects_data/theory/spec/los_specs/5ESN/output_00101/target_LOS/los2/peak_finding/spectrum/'
    write_spec(input_path,10)

if __name__ == '__main__':
    sys.exit(int(main() or 0))