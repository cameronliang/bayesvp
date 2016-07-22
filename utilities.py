import numpy as np

def gaussian_kernel(std):
    var = std**2
    size = 8*std +1 # this should gaurantee size to be odd.
    x = np.linspace(-100,100,size)
    norm = 1/(2*np.pi*var)
    return norm*np.exp(-(x**2/(2*std**2)))

def convolve_lsf(flux,lsf):
    # convolve 1-flux to remove edge effects wihtout using padding
    conv_flux = 1-np.convolve(1-flux,lsf,mode='same') /np.sum(lsf)
    return conv_flux
