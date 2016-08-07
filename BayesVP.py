import numpy as np
import time
import sys,os
import kombine

from Utilities import determine_autovp, print_config_params,\
					BIC_simple_estimate, printline 

def create_walkers_init(obs_spec):
	"""
	Initialize walkers with the free parameters

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

	c = 299792.458 # speed of light [km/s]
	temp_flags = obs_spec.vp_params_flags[~np.isnan(obs_spec.vp_params_flags)]
	n_params =  len(list(set(temp_flags)))

	# Get all the free parameters types
	final_vp_params_type = obs_spec.vp_params_type[~np.isnan(obs_spec.vp_params_flags)]

	p0 = np.zeros((n_params,obs_spec.nwalkers))
	for i in xrange(n_params):
		if final_vp_params_type[i] == 'logN':
			p0[i] = np.random.uniform(obs_spec.priors[0][0],
									obs_spec.priors[0][1],
									size=obs_spec.nwalkers)
		elif final_vp_params_type[i] == 'b':
			p0[i] = np.random.uniform(obs_spec.priors[1][0],
									obs_spec.priors[1][1],
									size=obs_spec.nwalkers)
		elif final_vp_params_type[i] == 'z':
			mean_z = obs_spec.priors[2][0]; 
			dv = obs_spec.priors[2][1];
			p0[i] = np.random.uniform(mean_z-dv/c,mean_z+dv/c,size=obs_spec.nwalkers)

	return np.transpose(p0)

def run_kombine_mcmc(obs_spec,chain_filename_ncomp):
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

	# define the MCMC parameters.
	p0 = create_walkers_init(obs_spec)
	ndim = np.shape(p0)[1]

	# Define the natural log of the posterior 
	lnprob = Posterior(obs_spec)

	# Set up the sampler
	sampler = kombine.Sampler(obs_spec.nwalkers, ndim, lnprob, processes=obs_spec.nthreads)

	# First do a rough burn in based on accetance rate.
	p_post_q = sampler.burnin(p0)
	p_post_q = sampler.run_mcmc(obs_spec.nsteps)
	
	np.save(chain_filename_ncomp + '.npy', sampler.chain)
	print("Written chain: %s.npy\n" % chain_filename_ncomp)
	printline()

def main(config_fname):
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

	# Determine if config is set to autovp 
	# If so, create multiple configs file specified by n_component_max
	auto_vp, n_component_min, n_component_max = determine_autovp(config_fname)

	if auto_vp:

		bic = np.zeros(n_component_max-n_component_min+1)

		for n in xrange(n_component_max-n_component_min+1):
			printline()
			print('Fitting %d components.. ' % (n + n_component_min))

			# Get new config filename; 
			config_fname_ncomp = (config_fname[:-4] + str(n+n_component_min)
								+ config_fname[-4:])

			# Load config parameter object 
			obs_spec = DefineParams(config_fname_ncomp)
			obs_spec.fileio_mcmc_params()
			obs_spec.fitting_data()
			obs_spec.fitting_params()
			obs_spec.spec_lsf()
			obs_spec.priors_and_init()

			#print_config_params(obs_spec)

			# Ouput filename for chain
			#run_kombine_mcmc(obs_spec,obs_spec.chain_fname)

			# Input the chian filename and number of data points
			bic[n] = BIC_simple_estimate(obs_spec.chain_fname,obs_spec)

			# compare with the previous fit 
			if n >= 1:
				# Stop fitting the previous bic is smaller (i.e better)
				if bic[n-1] <= bic[n]:
					components_count = np.arange(n_component_min,n_component_max)
					index = np.where(bic[:n+1] == np.min(bic[:n+1]))[0]
					
					printline()
					print('Based on BIC %d-component model is the best model' % components_count[index])
					printline()

					np.savetxt(obs_spec.mcmc_outputpath + '/bic.dat',np.c_[components_count[:n+1],bic[:n+1]],fmt=('%d','%.4f'),header='nComponents\tBICValues')
					
					break

	else:
		# Load config parameter object 
		obs_spec = DefineParams(config_fname)
		obs_spec.fileio_mcmc_params()
		obs_spec.fitting_data()
		obs_spec.fitting_params()
		obs_spec.spec_lsf()
		obs_spec.priors_and_init()
		
		print_config_params(obs_spec)
		
		# Run fit as specified in config
		run_kombine_mcmc(obs_spec,obs_spec.chain_fname)	

if __name__ == '__main__':

	from Config import DefineParams
	config_fname = sys.argv[1]
	main(config_fname)