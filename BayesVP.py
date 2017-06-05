################################################################################
#
# BayesVP.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Perform maximum likelihood fit to data given the config file specification 
# and output the MCMC chains     
################################################################################

import numpy as np
import sys
import os

from Config import DefineParams
from Utilities import determine_autovp,model_info_criterion,\
					  compare_model,gr_indicator,printline 

def _create_walkers_init(obs_spec):
	"""
	Initialize walkers with the free parameters. 
	Note that the initialization sets the number of parameters
	in the sampler. This initialization depends on the config file. 

	Parameters
	-----------
	obs_spec: obj
		Parameter object defined by the config file; Following 
		attributes are used in creating walkers initialization. 
	
		vp_params_flags: array_like
			Determines whether or not a parameter is fixed, 
			free or tied. If unique int -> free; if repeated 
			int -> tied; if None -> fixed.
		vp_params_type: array_like
			Determine the types of parameters (logN, b, z) for 
			initializationof the ranges.  
		nwalkers: int
			Number of walkers to run in MCMC.
	
	Returns
	-----------
	p0.T: array; shape = (nwalkers,ndim)
		The starting point of the walkers at n dimensions, where
		n = number of parameters. These will then walk/sample around 
		the parameters space based on the MCMC algorithm used. 
	"""

	temp_flags = obs_spec.vp_params_flags[~np.isnan(obs_spec.vp_params_flags)]
	n_params =  len(list(set(temp_flags)))

	# Get all the free parameters types
	final_vp_params_type = obs_spec.vp_params_type[~np.isnan(obs_spec.vp_params_flags)]

	p0 = np.zeros((n_params,obs_spec.nwalkers))
	for i in xrange(n_params):
		# require if-conditions for specific parameters 
		if final_vp_params_type[i] == 'logN':
			p0[i] = np.random.uniform(obs_spec.priors[0][0],
									obs_spec.priors[0][1],
									size=obs_spec.nwalkers)
		elif final_vp_params_type[i] == 'b':
			p0[i] = np.random.uniform(obs_spec.priors[1][0],
									obs_spec.priors[1][1],
									size=obs_spec.nwalkers)
		elif final_vp_params_type[i] == 'z':
			p0[i] = np.random.uniform(obs_spec.priors[2][0],
									obs_spec.priors[2][1],
									size=obs_spec.nwalkers)

	if obs_spec.cont_normalize:
		p1 = np.zeros((2,obs_spec.nwalkers))
		p1[0] = np.random.uniform(-1e-2,1e-2,size=obs_spec.nwalkers )# slope
		p1[1] = np.random.uniform(-1e-2,1e-2,size=obs_spec.nwalkers) # intercept

		p = np.concatenate((p0,p1),axis=0)
		return np.transpose(p)

	return np.transpose(p0)



def bvp_mcmc_single(obs_spec,chain_filename_ncomp = None):
	"""
	Run MCMC and save the chain based on parameters defined 
	in config file. 

	Parameters
	-----------
	obs_spec: obj
		Parameter object defined by the config file; Following 
		attributes are used in creating walkers initialization.
	chain_filename_ncomp: str
		Output filename without extention; '.npy' is assumed 
		added later

	Returns
	-----------
	chains: python format (.npy) binary file 
		One chain for the specified MCMC run. Use np.load
		to load the n-dim array into memory for manipulation.
	"""
	from Likelihood import Posterior

	if chain_filename_ncomp is None:
		chain_filename_ncomp = obs_spec.chain_fname

	# define the MCMC parameters.
	p0 = _create_walkers_init(obs_spec)
	ndim = np.shape(p0)[1]

	# Define the natural log of the posterior 
	lnprob = Posterior(obs_spec)

	if obs_spec.mcmc_sampler.lower() == 'emcee':
		import emcee
		sampler = emcee.EnsembleSampler(obs_spec.nwalkers, ndim, lnprob, 
										threads=obs_spec.nthreads) 
		sampler.run_mcmc(p0,obs_spec.nsteps)
		np.save(chain_filename_ncomp + '.npy', np.swapaxes(sampler.chain,0,1))
	elif obs_spec.mcmc_sampler.lower() == 'kombine':
		import kombine
		sampler = kombine.Sampler(obs_spec.nwalkers, ndim, lnprob, 
								  processes=obs_spec.nthreads)
		# First do a rough burn in based on accetance rate.
		p_post_q = sampler.burnin(p0)
		p_post_q = sampler.run_mcmc(obs_spec.nsteps)
		np.save(chain_filename_ncomp + '.npy', sampler.chain)
	else:
		print('No MCMC sampler selected.\n')
	print("Written chain: %s.npy\n" % chain_filename_ncomp)
	printline()

	# Compute Gelman-Rubin Indicator
	dnsteps = int(obs_spec.nsteps*0.05)
	n_steps = []; Rgrs = []	
	for n in xrange(dnsteps,obs_spec.nsteps):
		if n % dnsteps == 0:
			Rgrs.append(gr_indicator(sampler.chain[:n,:,:]))
			n_steps.append(n)
	n_steps = np.array(n_steps)
	Rgrs    = np.array(Rgrs)

	np.savetxt(chain_filename_ncomp+ '_GR.dat',np.c_[n_steps,Rgrs],fmt='%.5f',			   header='col1=steps\tcoln=gr_indicator')

	return 


def bvp_mcmc(config_fname):
	"""
	Run fixed number of component fit specified by the 
	config file or make copies of the configs and run up 
	to the maximum number of components until the best model 
	is found.

	Parameters:
	-----------
	config_fname: str
		Full path + the file name
		See ./readme.md for detaieded structure of the config file

	Returns
	-----------
	chains: python format (.npy) binary file 
		One or more chains for each MCMC run.  
	"""

	# Determine if config is set to automatically choose the best model 
	# If so, create multiple configs file specified by n_component_max
	auto_vp, n_component_min, n_component_max = determine_autovp(config_fname)

	if auto_vp:
		model_evidence = np.zeros(n_component_max-n_component_min+1)
		for n in xrange(n_component_max-n_component_min+1):
			printline()
			print('Fitting %d components.. ' % (n + n_component_min))

			# Get new config filename; 
			config_fname_ncomp = (config_fname[:-4] + str(n+n_component_min)
								+ config_fname[-4:])

			# Load config parameter object 
			obs_spec = DefineParams(config_fname_ncomp)

			# Run MCMC
			bvp_mcmc_single(obs_spec,obs_spec.chain_fname)

			# compute values of aic, bic or bf
			model_evidence[n] = model_info_criterion(obs_spec)
			components_count  = np.arange(n_component_min,n_component_max+1)

			# compare with the previous fit 
			if 0 < n <= n_component_max-1:

				if obs_spec.model_selection.lower() in ('odds','bf'):   
					index = np.where(model_evidence[:n+1] == np.max(model_evidence[:n+1]))[0]

				elif obs_spec.model_selection.lower() in ('aic','bic'):
					index = np.where(model_evidence[:n+1] == np.min(model_evidence[:n+1]))[0]

				# Compars BIC/AIC/Odds ratios
				if compare_model(model_evidence[n-1], model_evidence[n],obs_spec.model_selection):
					printline()
					print('Based on %s: %d-component model is the best model' % (obs_spec.model_selection,components_count[index]))
					printline()
					np.savetxt(obs_spec.mcmc_outputpath + '/' + obs_spec.model_selection +'_'+obs_spec.chain_short_fname[:-1]+'.dat',
					np.c_[components_count[:n+1],model_evidence[:n+1]],fmt=('%d','%.4f'),header='nComponents\tValues')
					break
	else:
		# Load config parameter object 
		obs_spec = DefineParams(config_fname)		
		obs_spec.print_config_params()
		
		# Run fit as specified in config
		bvp_mcmc_single(obs_spec)	

if __name__ == '__main__':

	config_fname = sys.argv[1]
	bvp_mcmc(config_fname)
	