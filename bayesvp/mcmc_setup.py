################################################################################
#
# mcmc_setup.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Perform maximum likelihood fit to data given the config file specification 
# and output the MCMC chains     
################################################################################


import numpy as np
import sys
import os


from bayesvp.config import DefineParams
from bayesvp.utilities import determine_autovp,model_info_criterion,\
					  compare_model,gr_indicator 


def _create_walkers_init(config_params):
	"""
	Initialize walkers with the free parameters. 
	Note that the initialization sets the number of parameters
	in the sampler. This initialization depends on the config file. 

	Parameters
	-----------
	config_params: obj
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
	p0.T: array; shape = (nwalkers, ndim)
		The starting point of the walkers at n dimensions, where
		n = number of parameters. These will then walk/sample 
		around the parameters space based on the MCMC algorithm 
		used.
	"""

	temp_flags = config_params.vp_params_flags[~np.isnan(config_params.vp_params_flags)]
	n_params =  len(list(set(temp_flags)))

	# Get all the free parameters types
	final_vp_params_type = config_params.vp_params_type[~np.isnan(config_params.vp_params_flags)]

	p0 = np.zeros((n_params,config_params.nwalkers))
	for i in xrange(n_params):
		# Match priors with the parameter types
		if final_vp_params_type[i] == 'logN':
			p0[i] = np.random.uniform(config_params.priors[0][0],
									config_params.priors[0][1],
									size=config_params.nwalkers)
		elif final_vp_params_type[i] == 'b':
			p0[i] = np.random.uniform(config_params.priors[1][0],
									config_params.priors[1][1],
									size=config_params.nwalkers)
		elif final_vp_params_type[i] == 'z':
			p0[i] = np.random.uniform(config_params.priors[2][0],
									config_params.priors[2][1],
									size=config_params.nwalkers)

	if config_params.cont_normalize:
		p1 = np.zeros((config_params.cont_nparams,config_params.nwalkers))

		for i in range(config_params.cont_nparams):
			p1[i] = np.random.uniform(-1,1,size=config_params.nwalkers )

		p = np.concatenate((p0,p1),axis=0)
		return np.transpose(p)

	return np.transpose(p0)



def bvp_mcmc_single(config_params,chain_filename_ncomp = None):
	"""
	Run MCMC and save the chain based on parameters defined 
	in config file. 

	Parameters
	-----------
	config_params: obj
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
	from bayesvp.likelihood import Posterior

	if chain_filename_ncomp is None:
		chain_filename_ncomp = config_params.chain_fname

	# define the MCMC parameters.
	p0 = _create_walkers_init(config_params)
	ndim = np.shape(p0)[1]


	# Define the natural log of the posterior 
	lnprob = Posterior(config_params)

	if config_params.mcmc_sampler.lower() == 'emcee':
		import emcee
		sampler = emcee.EnsembleSampler(config_params.nwalkers, ndim, lnprob, 
										threads=config_params.nthreads)
		sampler.run_mcmc(p0,config_params.nsteps)
		np.save(chain_filename_ncomp + '.npy', np.swapaxes(sampler.chain,0,1))
	
	elif config_params.mcmc_sampler.lower() == 'kombine':
		import kombine
		sampler = kombine.Sampler(config_params.nwalkers, ndim, lnprob,
								  processes=config_params.nthreads)
		
		# First do a rough burn in based on accetance rate.
		p_post_q = sampler.burnin(p0)
		p_post_q = sampler.run_mcmc(config_params.nsteps)
		np.save(chain_filename_ncomp + '.npy', sampler.chain)
	
	else:
		sys.exit('Error! No MCMC sampler selected.\nExiting program...')


	# Compute Gelman-Rubin Indicator
	dnsteps = int(config_params.nsteps*0.05)
	n_steps = []; Rgrs = []	
	for n in xrange(dnsteps,config_params.nsteps):
		if n % dnsteps == 0:
			Rgrs.append(gr_indicator(sampler.chain[:n,:,:]))
			n_steps.append(n)
	n_steps = np.array(n_steps)
	Rgrs    = np.array(Rgrs)

	np.savetxt(chain_filename_ncomp+ '_GR.dat',np.c_[n_steps,Rgrs],fmt='%.5f',
				header='col1=steps\tcoln=gr_indicator')

	return 


def bvp_mcmc(config_fname,print_config=False):
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

			# Get new config filename; 
			basename_with_path, config_extension = os.path.splitext(config_fname)
			config_fname_ncomp = (basename_with_path + str(n+n_component_min)
								+ config_extension)

			# Load config parameter object 
			config_params = DefineParams(config_fname_ncomp)

			# Run MCMC
			bvp_mcmc_single(config_params,config_params.chain_fname)

			# compute values of aic, bic or bf
			model_evidence[n] = model_info_criterion(config_params)
			components_count  = np.arange(n_component_min,n_component_max+1)

			# compare with the previous fit 
			if 0 < n <= n_component_max-1:

				#if config_params.model_selection.lower() in ('odds','bf'):   
				#	index = np.where(model_evidence[:n+1] == np.max(model_evidence[:n+1]))[0]

				if config_params.model_selection.lower() in ('aic','bic'):
					index = np.where(model_evidence[:n+1] == np.min(model_evidence[:n+1]))[0]

				# Compare BIC/AIC/Odds ratios
				if compare_model(model_evidence[n-1], model_evidence[n],config_params.model_selection):
					np.savetxt(config_params.mcmc_outputpath  + '/' + 
								config_params.model_selection + '_' + 
								config_params.chain_short_fname[:-1]+'.dat',
								np.c_[components_count[:n+1],model_evidence[:n+1]],
								fmt=('%d','%.4f'),header='nComponents\tValues')
					break
	else:
		# Load config parameter object 
		config_params = DefineParams(config_fname)

		if print_config:
			config_params.print_config_params()
		
		# Run fit as specified in config
		bvp_mcmc_single(config_params)	
