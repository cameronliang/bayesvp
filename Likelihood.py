################################################################################
#
# Likelihood.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Defined the likelihood and posterior distribution for the model given the 
# data.      
################################################################################

import numpy as np

from Model import generic_prediction

np.seterr(all='ignore') # Ignore floating point warnings.

def tophat_prior(model_x, x_left,x_right):
	if model_x >= x_left and model_x < x_right:
		return 0
	else:
		return -np.inf

class Posterior(object):
    """
    Define the natural log of the posterior distribution

    Parameters:
    ----------- 
    obs_spec: 
        Parameters object defined by the config file
    
    Returns:
    -----------
    lnprob: function
        The posterior distribution as a function of the 
        input parameters given the spectral data
    """
    def __init__(self,obs_spec):
        self.obs_spec = obs_spec

    def lnlike(self,alpha):
        """
        Likelihood assumes flux follow a Gaussian
        
        Returns
        --------
        ln_likelihood: float
            Natural log of the likelihood
        """

        # Flux of the model
        model_flux = generic_prediction(alpha,self.obs_spec)
        
        resid = self.obs_spec.flux - model_flux

        # Natural log of gaussian likelihood with normalization included 
        ln_likelihood = np.sum(0.5*np.log(2*np.pi*self.obs_spec.dflux**2)             -0.5*resid**2/self.obs_spec.dflux**2)

        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood

    def lnprior(self,alpha):
        """
        Natural Log of the priors for three types of 
        parameters [logN, b, z]

        Note that for redshift z, the ranges are defined 
        to be [mean_z-v/c,mean_z+v/c]  
        """
        
        # Define these for clarity
        min_logN,max_logN = self.obs_spec.priors[0] 
        min_b,max_b       = self.obs_spec.priors[1]
        min_z,max_z         = self.obs_spec.priors[2]

        # Select the parameters that are free or tie (i.e not fixed)
        final_vp_params_type = self.obs_spec.vp_params_type[~np.isnan(self.obs_spec.vp_params_flags)]

        sum_logN_prior = 0; sum_b_prior = 0; sum_z_prior = 0;

        model_redshifts = []
        for i in xrange(len(alpha)):
            if final_vp_params_type[i] == 'logN':
                sum_logN_prior += tophat_prior(alpha[i],min_logN,max_logN)
            elif final_vp_params_type[i] == 'b':
                sum_b_prior += tophat_prior(alpha[i],min_b, max_b)
            elif final_vp_params_type[i] == 'z':
                model_redshifts.append(alpha[i])
                sum_z_prior += tophat_prior(alpha[i],min_z,max_z)
        
        # Make sure multiple components do not swap
        model_redshifts = np.array(model_redshifts)
        if not all(sorted(model_redshifts) == model_redshifts):
            sum_z_prior = -np.inf 

        return sum_logN_prior + sum_b_prior + sum_z_prior

    def __call__(self,alpha):
        """
        Posterior distribution

        Returns
        ---------
        lnprob: float
            Natural log of posterior probability
        """
        lp = self.lnprior(alpha)
        
        if np.isinf(lp):
            return -np.inf
        else:
            return np.atleast_1d(lp + self.lnlike(alpha))[0]
