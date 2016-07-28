import numpy as np
import time
import sys,os

from Utilities import determine_autovp, print_config_params,\
					BIC_gaussian_kernel,BIC_best_estimator,\
					printline

def walkers_init():
	fname = './data/walkers_init.dat'
	init_ranges = np.loadtxt(fname,unpack=True,usecols=[1,2])
	return np.transpose(init_ranges)

def create_walkers_init(obs_spec):
	"""
	Initialize walkers with the free parameters 

	Parameters
	-----------
	flags
	vp_params_type
	nwalkers
	"""

	c = 299792.458
	temp_flags = obs_spec.vp_params_flags[~np.isnan(obs_spec.vp_params_flags)]
	n_params =  len(list(set(temp_flags)))

	# Get all the free parameters types
	final_vp_params_type = obs_spec.vp_params_type[~np.isnan(obs_spec.vp_params_flags)]

	# Get walkers initialization ranges defined by ./data/walkers_init.dat
	priors = walkers_init()

	p0 = np.zeros((n_params,obs_spec.nwalkers))
	for i in range(n_params): 
		if final_vp_params_type[i] == 'logN':
			p0[i] = np.random.uniform(priors[0][0],priors[0][1],size=obs_spec.nwalkers)
		elif final_vp_params_type[i] == 'b':
			p0[i] = np.random.uniform(priors[1][0],priors[1][1],size=obs_spec.nwalkers)
		elif final_vp_params_type[i] == 'z':
			mean_z = priors[2][0]; dv = priors[2][1];
			p0[i] = np.random.uniform(mean_z-dv/c,mean_z+dv/c,size=obs_spec.nwalkers)

	return np.transpose(p0)

def run_kombine_mcmc(obs_spec,chain_filename_ncomp):

	from Likelihood import posterior
	import kombine

	# define the MCMC parameters.
	p0 = create_walkers_init(obs_spec)
	ndim = np.shape(p0)[1]

	# Define the natural log of the posterior 
	lnprob = posterior(obs_spec)

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
	Below is the skeleton of the main
	Create copies of config file to run the fit with 
	multiple similar to CLOUDY 
	"""

	# Determine if config is set to autovp 
	# If so, create multiple configs file specified by n_component_max
	auto_vp, n_component_max = determine_autovp(config_fname)

	if auto_vp:
		bic = np.zeros(n_component_max)

		for n in xrange(n_component_max):
			print('Fitting %d components.. ' % (n + 1))
			# Get new config filename; 
			dot_index =  config_fname.find('.')
			config_fname_ncomp = (config_fname[:dot_index] + str(n+1)
							 + config_fname[dot_index:])

			
			# Load config parameter object 
			obs_spec = obs_data(config_fname_ncomp)
			obs_spec.fileio_mcmc_params()
			obs_spec.fitting_data()
			obs_spec.fitting_params()
			obs_spec.spec_lsf()

			print_config_params(obs_spec)
			# Ouput filename for chain
			chain_filename_ncomp = obs_spec.chain_fname +  str(n+1)			
			run_kombine_mcmc(obs_spec,chain_filename_ncomp)
			
			# Input the chian filename and number of data points
			bic[n] = BIC_best_estimator(chain_filename_ncomp,len(obs_spec.flux))

			# compare with the previous fit 
			if n >= 1:
				# Stop fitting the previous bic is smaller (i.e better) 
				if bic[n-1] <= bic[n]:
					print("Based on BIC, %d component model is the best fit." % (n))					 
					break

	else:
		# Load config parameter object 
		obs_spec = obs_data(config_fname)
		obs_spec.fileio_mcmc_params()
		obs_spec.fitting_data()
		obs_spec.fitting_params()
		obs_spec.spec_lsf()

		print_config_params(obs_spec)

		# Run fit as specified in config
		run_kombine_mcmc(obs_spec,obs_spec.chain_fname)	

if __name__ == '__main__':

	from Config import obs_data
	config_fname = sys.argv[1]
	main(config_fname)