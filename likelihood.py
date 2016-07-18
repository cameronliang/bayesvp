import numpy as np
import pylab as pl

from observation import obs_spec
from model import model_prediction,generic_prediction
np.seterr(all='ignore')

# speed of light [km/s]
c = 299792.458

# Read in hardwired priors
priors = np.transpose(np.loadtxt('./data/priors.dat',
         unpack=True,usecols=[1,2]))

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

    final_vp_params_type = obs_spec.vp_params_type[~np.isnan(obs_spec.vp_params_flags)]

    sum_logN_prior = 0; sum_b_prior = 0; sum_z_prior = 0;

    z_priors = []
    for i in range(len(alpha)):
        if final_vp_params_type[i] == 'logN':
            sum_logN_prior += tophat_prior(alpha[i],priors[0][0],priors[0][1])
        elif final_vp_params_type[i] == 'b':
            sum_b_prior += tophat_prior(alpha[i],priors[1][0],priors[1][1])
        elif final_vp_params_type[i] == 'z':
            z_priors.append(alpha[i])
            sum_z_prior += tophat_prior(alpha[i],priors[2][0]-priors[2][1]/c,priors[2][0] + priors[2][1]/c)
    
    z_priors = np.array(z_priors)
    if all(sorted(z_priors) == z_priors):
        pass 
    else:
        sum_z_prior = -np.inf

    return sum_logN_prior + sum_b_prior + sum_z_prior

def lnprob(alpha):
    lp = lnprior(alpha)
    if np.isinf(lp):
        return -np.inf
    else:
        return np.atleast_1d(lp + lnlike(alpha))[0]
