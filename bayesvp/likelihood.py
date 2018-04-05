################################################################################
#
# likelihood.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Defined the likelihood and posterior distribution for the model given the 
# data.      
################################################################################

import numpy as np

from bayesvp.vp_model import continuum_model_flux

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
    config_params: 
        Parameters object defined by the config file
    
    Returns:
    -----------
    lnprob: function
        The posterior distribution as a function of the 
        input parameters given the spectral data
    """
    def __init__(self,config_params):
        self.config_params = config_params

    def lnlike(self,alpha):
        """
        Likelihood assumes flux follow a Gaussian
        
        Returns
        --------
        ln_likelihood: float
            Natural log of the likelihood
        """

        # Flux of the model
        model_flux = continuum_model_flux(alpha,self.config_params)
        
        resid = self.config_params.flux - model_flux

        # Natural log of gaussian likelihood with normalization included 
        ln_likelihood = np.sum(0.5*np.log(2*np.pi*self.config_params.dflux**2)
                              -0.5*resid**2/self.config_params.dflux**2)

        if np.isnan(ln_likelihood):
            return -np.inf
        else:
            return ln_likelihood

    def lnprior(self,alpha):
        """
        Natural Log of the priors for three types of 
        parameters [logN, b, z]

        Note that for redshift z, the ranges are defined 
        to be [mean_z-v/c, mean_z+v/c]

        Note also the default continuum parameters
        are limited +/- 1. It is so because 'pivoting'
        is used during the fit. The value can be set
        by cont_prior flag in config file (cont_flag 2.0)
        """

        # Define these for clarity
        min_logN,max_logN = self.config_params.priors[0] 
        min_b,max_b       = self.config_params.priors[1]
        min_z,max_z       = self.config_params.priors[2]

        cont_priors      = self.config_params.cont_prior # array_like

        # Select the parameters that are free or tie (i.e not fixed)
        final_vp_params_type = self.config_params.vp_params_type[~np.isnan(self.config_params.vp_params_flags)]

        sum_logN_prior = 0; sum_b_prior = 0; sum_z_prior = 0

        model_redshifts = []
        
        for i in xrange(len(final_vp_params_type)):
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

        total_prior = sum_logN_prior + sum_b_prior + sum_z_prior

        # linear continuum slope and intercept priors
        if self.config_params.cont_normalize:
            contiuum_prior = 0
            for i in range(1,self.config_params.cont_nparams+1):
                contiuum_prior += tophat_prior(alpha[-i],-cont_priors[i-1],
                                                cont_priors[i-1])
            total_prior = total_prior + contiuum_prior

        return total_prior

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
