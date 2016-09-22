################################################################################
#
# Model.py   		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Voigt profile model; Produce prediction of model given the config file   
# (specifically the initial parameters). It can be imported to produce 
# a voigt profile on demand specified by user as well. 
################################################################################

import numpy as np
from scipy.special import wofz
import sys
import os

from Utilities import convolve_lsf,get_transitions_params

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

def WavelengthArray(wave_start, wave_end, dv):
    """
    Create Wavelength array with resolution dv
    
    Parameters:
    ----------
    wave_start: float
        starting wavelength of array [\AA]
    wave_end: float
        ending wavelength of array [\AA]
    dv: float
        resolution element [km/s]
    
    Returns
    ----------
    wavelength_array: array_like
    """ 
    
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

def Voigt(b, z, nu, nu0, Gamma):
    """
    Generate Voigt Profile for a given transition

    Parameters:
    ----------
    b: float
        b parameter of the voigt profile
    z: float
        resfhit of the absorption line
    nu: array_like
        rest frame frequncy array
    nu0: float
        rest frame frequency of transition [1/s]
    Gamma: float
        Damping coefficient (transition specific)

    Returns:
    ----------
    V: array_like
        voigt profile as a function of frequency
    """

    delta_nu = nu - nu0 / (1+z)
    delta_nuD = b * nu / c
    
    prefactor = 1.0 / ((np.pi**0.5)*delta_nuD)
    x = delta_nu/delta_nuD
    a = Gamma/(4*np.pi*delta_nuD)

    return prefactor * voigt(x,a)  

def bParameter(logT, b_nt, mass):
    """
    Combined thermal and non-thermal velocity

    Parameters:
    ----------
    mass: float
        mass of ion in the transition [grams]
    logT: array_like
        log10 of Temperature [Kelvin]
    b_nt: array_like
        non-thermal velocity dispersion [km/s]

    Returns:
    ----------
    b parameter: array_like; [km/s]
    """
    
    temp = 10**logT
    b_thermal = np.sqrt(2. * kB*temp / mass)*cm_km  # [km/s]
    return np.sqrt(b_thermal**2 + b_nt**2)


def General_Intensity(logN, b, z, wave, atomic_params):
    """
    Takes a general combination of atomic 
    parameters, without specifying the name of the transition 
    to compute the flux
    
    Parameters:
    ----------
    logN: float
        log10 of column density [cm-2] 
    b: float
        total b parameter [km/s]
    z: float
        redshift of system
    wave: 1D array
        observed wavelength array
    atomic_params: array
        array of oscillator strength, rest frame wavelength [\AA], damping coefficient, mass [grams] of the transition

    Returns:
    ----------
    intensity: array
        normalized flux of the spectrum    
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

def simple_spec(logN, b, z, wave, atom=None, state=None, lsf=1):
    """
    Generate a single component absorption for all transitions
    within the given observed wavelength, redshift for the desired
    atom and state

    Parameters:
    ----------
    logN: float
        log10 of column density [cm-2] 
    b: float
        total b parameter [km/s]
    z: float
        redshift of system
    wave: 1D array
        observed wavelength array

    Returns:
    ----------
    intensity: array
        normalized flux of the spectrum    
    """

    if atom is None or state is None:
        print('Enter atom and ionization state (e.g., C IV)\n')
        exit()

    atomic_params = get_transitions_params(atom,state,min(wave),max(wave),z)
    n_transitions = len(atomic_params)
    
    spec = []
    for l in xrange(n_transitions):
        if not np.isnan(atomic_params[l]).any():
            model_flux = General_Intensity(logN,b,z,wave,atomic_params[l]) 
            spec.append(convolve_lsf(model_flux,lsf)) 
        else:
            return np.ones(wave)
    # Return the convolved model flux with LSF
    return np.product(spec,axis=0)

def generic_prediction(alpha, obs_spec_obj):
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
    
    Returns:
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
    wave = WavelengthArray(1200,1230,2)

    flux = simple_spec(15,20,0,wave)
    #print flux
    pl.step(wave,flux)
    pl.xlim([1214,1217])
    pl.ylim([0,1.3])
    pl.show()