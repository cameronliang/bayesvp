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
    model_flux = generic_prediction(alpha,obs_spec)

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
        sum_logN_prior += tophat_prior(logN_priors[n],0,21)
        sum_b_prior += tophat_prior(b_priors[n],0,100)
        sum_z_prior += tophat_prior(z_priors[n],-1000/c,1000/c)
 
    return sum_logN_prior + sum_b_prior + sum_z_prior

def generic_lnprior(alpha):

    final_vp_params_type = obs_spec.vp_params_type[~np.isnan(obs_spec.vp_params_flags)]

    sum_logN_prior = 0; sum_b_prior = 0; sum_z_prior = 0;
    
    z_priors = []
    for i in range(len(alpha)):
        if final_vp_params_type[i] == 'logN':
            sum_logN_prior += tophat_prior(alpha[i],0,21)
        elif final_vp_params_type[i] == 'b':
            sum_b_prior += tophat_prior(alpha[i],0,100)
        elif final_vp_params_type[i] == 'z':
            z_priors.append(alpha[i])
            sum_z_prior += tophat_prior(alpha[i],-1000/c,1000/c)
    
    z_priors = np.array(z_priors)
    if all(sorted(z_priors) == z_priors):
        pass 
    else:
        sum_z_prior = -np.inf

    return sum_logN_prior + sum_b_prior + sum_z_prior

def lnprob(alpha):

    #lp = lnprior(alpha)
    lp = generic_lnprior(alpha)

    if np.isinf(lp):
        return -np.inf
    else:
        return np.atleast_1d(lp + lnlike(alpha))[0]
