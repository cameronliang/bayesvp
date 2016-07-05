import numpy as np
import pylab as pl

from observation import obs_spec
from model import model_prediction,generic_prediction
np.seterr(all='ignore')


c = 299792.458

def tophat_prior(model_x, x_left,x_right):
	if model_x >= x_left and model_x < x_right:
		return 0
	else:
		return -np.inf

def lnlike(alpha):
    """
    Likelihood assumes the uncertaities of flux follow
    a gaussian distribution
    """
    # flux of the model
    model_flux = generic_prediction(alpha,obs_spec.wave,obs_spec.transitions_params_array)
    
    resid = obs_spec.flux - model_flux
    ln_likelihood = np.sum(0.5*np.log(2*np.pi*obs_spec.dflux**2)             -0.5*resid**2/obs_spec.dflux**2)

    if np.isnan(ln_likelihood):
 		return -np.inf
    else:
        return ln_likelihood

def lnprior(alpha):

    logN_priors = alpha[::3]; b_priors    = alpha[1::3]
    z_priors = alpha[2::3]
    sum_logN_prior = 0; sum_b_prior = 0; 
    
    if all(sorted(z_priors) == z_priors):
        sum_z_prior = 0; 
    else:
        sum_z_prior = -np.inf
    for n in xrange(obs_spec.n_component):
        sum_b_prior += tophat_prior(b_priors[n],0,100)
        sum_logN_prior += tophat_prior(logN_priors[n],0,21)
        sum_z_prior += tophat_prior(z_priors[n],-1000/c,1000/c)
 
    return sum_logN_prior + sum_b_prior + sum_z_prior

def lnprob(alpha):

    lp = lnprior(alpha)
    if np.isinf(lp):
        return -np.inf
    else:
        return np.atleast_1d(lp + lnlike(alpha))[0]
