import numpy as np
import time
import sys,os

from Config import obs_spec

def walkers_init():
	fname = './data/walkers_init.dat'
	init_ranges = np.loadtxt(fname,unpack=True,usecols=[1,2])
	return np.transpose(init_ranges)

def create_walkers_init(vp_params,vp_params_type,flags):
	"""
	Initialize walkers with the free parameters 
	"""

	c = 299792.458
	temp_flags = flags[~np.isnan(flags)]
	n_params =  len(list(set(temp_flags)))

	# Get all the free parameters types
	final_vp_params_type = vp_params_type[~np.isnan(flags)]

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

def run_kombine(output_fname):
	from Likelihood import lnprob
	import kombine

	# define the MCMC parameters.
	p0 = create_walkers_init(obs_spec.vp_params,
		 obs_spec.vp_params_type,obs_spec.vp_params_flags)
	
	ndim = np.shape(p0)[1]

	# Set up the sampler
	print("Running MCMC...")
	print("Number of parameters = %i" % ndim)
	sampler = kombine.Sampler(obs_spec.nwalkers, ndim, lnprob, processes=obs_spec.nthreads)

	# First do a rough burn in based on accetance rate.
	p_post_q = sampler.burnin(p0)
	p_post_q = sampler.run_mcmc(obs_spec.nsteps)
	
	np.save(output_fname + '.npy', sampler.chain)
	print("Written chain: %s.npy" % output_fname)

if __name__ == '__main__':

	p0 = create_walkers_init(obs_spec.vp_params,obs_spec.vp_params_type,obs_spec.vp_params_flags)

	run_kombine(obs_spec.chain_fname)