import numpy as np
from scipy.special import wofz
import sys
import os

from Utilities import convolve_lsf

# Fundamental constant [cgs units]
h  = 6.6260755e-27   # planck constants
kB = 1.380658e-16    # Boltzmann constant
c  = 2.99792458e10   # speed of light
m  = 9.10938291e-28  # electron mass
e  = 4.80320425e-10  # electron charge
mH = 1.67e-24        # proton mass
sigma0 = 0.0263      # Cross section [cm^2/sec]

# Units Conversion
cm_km = 1.e-5  # Convert cm to km
km_cm = 1.e5   # Convert km to cm
ang_cm = 1.e-8 # Convert Angstrom to cm
kpc_cm = 3.0856e21 # Convert kpc to cm 


class Transition:
    def __init__(self,name,wave,osc_f,gamma,mass):
        self.name  = name
        self.wave  = wave
        self.osc_f = osc_f
        self.gamma = gamma
        self.mass  = mass

def WriteSpec_ascii(wavelength,flux,fname):
    """Write Spectrum in ascii format"""
    f = open(fname,'w')
    for i in range(len(wavelength)):
        f.write('%f\t%f\n' % (wavelength[i],flux[i]))
    f.close()

def ReadTransitionData():    
    amu = 1.66053892e-24   # 1 atomic mass in grams
    data_path = os.path.dirname(__file__) # Absolute path for BayseVP
    data_file = data_path + '/data/atomic_data.dat'
    name  = np.loadtxt(data_file, dtype=str, usecols=[0])
    wave,osc_f,gamma,mass  = np.loadtxt(data_file, unpack=True,
                                        usecols=[1,2,3,4])
    mass = mass*amu

    transition_dict = {}
    for i in xrange(len(name)):
        transition_dict[str(name[i])] = Transition(name[i],wave[i],osc_f[i],gamma[i],mass[i])
    return transition_dict

def WavelengthArray(wave_start,wave_end,dv):
    """Create Wavelength array with resolution = dv""" 
    
    c = 299792.458       # speed of light  [km/s]
    wave_start = float(wave_start); wave_end = float(wave_end)

    # Calcualte Total number of pixel given the resultion and bounds of spectrum
    total_number_pixel = np.int(np.log10(wave_end/wave_start) / np.log10(1 + dv/c) + 0.5)
    array_index        = np.arange(0,total_number_pixel,1)
    
    # Return wavelength array
    return wave_start * ((1 + dv/c)**array_index)

def voigt(x, a):
    """
    Real part of Faddeeva function, where    
    w(z) = exp(-z^2) erfc(jz)
    """
    z = x + 1j*a
    return wofz(z).real

def Voigt(b,z,nu,nu0,Gamma):
    """
    Generate Voigt Profile for a given transition
    Input: 

    1. Damping coefficient: Gamma (transition specific )
    2. wave0  = rest frame wavelength
    3. wavelength = wavelength array for the spectrum. 
    4. Temperature of the gas

    Returns:
    ----------
    voigt profile as a function of frequency
    """

    delta_nu = nu - nu0 / (1+z)
    delta_nuD = b * nu / c
    
    prefactor = 1.0 / ((np.pi**0.5)*delta_nuD)
    x = delta_nu/delta_nuD
    a = Gamma/(4*np.pi*delta_nuD)

    return prefactor * voigt(x,a)  

def bParameter(logT,b_turb,mass):
    """
    mass   = mass of ion in the transition [grams]
    logT   = log10 of Temperature          [Kelvin]
    b_turb = turbulance velocity           [km/s]

    Returns:
    ----------
    b parameter: array_like; [km/s]
        Combined thermal and non-thermal velocity
    """
    
    temperature = 10**logT
    b_thermal = np.sqrt(2. * kB*temperature / mass)*cm_km  # units: [km/s]
    return np.sqrt(b_thermal**2 + b_turb**2)

def Intensity(logN,b,z,wave,transition_name):
    """
    This function is sufficient in itself to 
    create an absorption line. 

    Parameters:
    ---------- 
        logN = log10 column density               [cm-2]
        b    = b parameter                        [km/s]
        z    = redshift of the gas                 
        wave = input rest frame wavelength array  [\AA]

    Returns:
    ----------
    Intensity: array_like
        Normlized Intensity with length = len(wave)
    """    
    transition_data = ReadTransitionData()

    # Obtain transition data
    f       = transition_data[transition_name].osc_f
    wave0   = transition_data[transition_name].wave
    gamma   = transition_data[transition_name].gamma

    # Convert to cgs units
    b       = b * km_cm       # Convert km/s to cm/s
    N       = 10**logN        # Column densiy in linear space
    lambda0 = wave0*ang_cm    # Convert Angstrom to cm
    nu0     = c/lambda0       # Rest frame frequency 
    nu      = c/(wave*ang_cm) # Frequency array 

    # Compute Optical depth
    tau = N*sigma0*f*Voigt(b,z,nu,nu0,gamma)
    
    # Normalized intensity
    return np.exp(-tau.astype(np.float))

def General_Intensity(logN,b,z,wave,atomic_params):
    """
    This function takes a general combination of atomic 
    parameters, without specifying the name of the transition.
    """
    f,wave0,gamma,mass = atomic_params

    # Convert to cgs units
    b       = b * km_cm       # Convert km/s to cm/s
    N       = 10**logN        # Column densiy in linear space
    lambda0 = wave0*ang_cm    # Convert Angstrom to cm
    nu0     = c/lambda0       # Rest frame frequency 
    nu      = c/(wave*ang_cm) # Frequency array 

    # Compute Optical depth
    tau = N*sigma0*f*Voigt(b,z,nu,nu0,gamma)

    # Return Normalized intensity
    return np.exp(-tau.astype(np.float))

def model_prediction_components(alpha,wave,n_component,transition_name):
    if n_component == 1:
        logN,b,z = alpha
        return Intensity(logN,b,z,wave,transition_name)
    
    elif n_component > 1:
        spec = np.ones((n_component,len(wave)))
        alpha = alpha.reshape(n_component,3)
        for i in xrange(n_component):
            logN,b,z = alpha[i]
            spec[i] = Intensity(logN,b,z,wave,transition_name)
        return np.product(spec,axis=0)
    else:
        print("The number of component cannot be 0 or negative\n")
        exit()    

def model_prediction(alpha,wave,n_component,transition_names):
    
    n_trans = len(transition_names)
    spec = np.ones((n_trans,len(wave)))
    for i in xrange(n_trans):
        spec[i] = model_prediction_components(alpha,wave,n_component,transition_names[i])
    return np.product(spec,axis=0)

def generic_prediction(alpha,obs_spec_obj):
    """
    This is a model flux produced by a generic combination 
    of the voigt profile parameters by arbitrary combination
    of fixing/tieing/freeing any parameters; see Config.py

    Parameters:
    ----------
    alpha: array_like
        One dimensional array of n parameters; The structure 
        of the array needs to match the ones specified in the 
        config file  
    obs_spec_obj:
        Paramters object defined by the config file
        see ./Config.py
    Returns
    ----------
    Model flux: 1D array;
        Predicted flux based on the paramters with length equal to 
        the length of the input wavelength array
    """
    component_flags = obs_spec_obj.vp_params_flags.reshape(obs_spec_obj.n_component,3)

    spec = [] 
    for i in xrange(obs_spec_obj.n_component):
        # Re-group parameters intro [logN, b,z] for each component
        temp_alpha = np.zeros(3)
        for j in xrange(3):
            # NaN indicates parameter has been fixed
            if np.isnan(component_flags[i][j]): 
                # access the fixed value from vp_params after removing the upper case letter 
                temp_alpha[j] = float(obs_spec_obj.vp_params[i][j][:-1])
            else: 
                # access the index map from flags to alpha 
                temp_alpha[j] = alpha[component_flags[i][j]] 


        # Compute spectrum for each component, region, and transition.
        n_wavelength_regions = len(obs_spec_obj.wave_begins)
 
        for k in xrange(n_wavelength_regions):  
            n_transitions = len(obs_spec_obj.transitions_params_array[i][k])
            for l in xrange(n_transitions):
                if not np.isnan(obs_spec_obj.transitions_params_array[i][k][l]).any():
                    model_flux = General_Intensity(temp_alpha[0],temp_alpha[1],temp_alpha[2],obs_spec_obj.wave,obs_spec_obj.transitions_params_array[i][k][l])

                    # Convolve (potentially )LSF for each region 
                    spec.append(convolve_lsf(model_flux,obs_spec_obj.lsf[k])) 

    # Return the convolved model flux with LSF
    return np.product(spec,axis=0)

if __name__ == '__main__':

    import matplotlib.pyplot as pl
    from Config import DefineParams
    data_path = os.path.dirname(os.path.abspath(__file__))
    config_fname = data_path + '/tests/config_CIV.dat'

    obs_spec = DefineParams(config_fname)
    obs_spec.fileio_mcmc_params()
    obs_spec.fitting_data()
    obs_spec.fitting_params()
    obs_spec.spec_lsf()
    
    alpha = np.array([13.766,14.42,-0.000139]) 
    flux = generic_prediction(alpha,obs_spec)
    
    pl.step(obs_spec.wave,obs_spec.flux,c='k')
    pl.step(obs_spec.wave,obs_spec.dflux,c='r')
    pl.plot(obs_spec.wave,flux,lw=1.5,c='b')
    pl.ylim([0,1.4])
    pl.show()
