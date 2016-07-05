import numpy as np
import time
import sys,os

from observation import obs_spec

def create_guass_init(vp_params):
	c = 299792.458
	n_component,n_params_component = vp_params.shape
	n_params = int(n_component*n_params_component)

	vp_params_flat = obs_spec.vp_params.flatten()
	p0 = np.zeros((n_params,obs_spec.nwalkers))

	p0[::3] = np.random.uniform(5,20,size=obs_spec.nwalkers)
	p0[1::3] = np.random.uniform(0,30,size=obs_spec.nwalkers)
	p0[2::3] = np.random.uniform(-50/c,50/c,size=obs_spec.nwalkers)

	#for i in xrange(n_params):
		#p0[i] = np.random.normal(vp_params_flat[i],0.01,size=obs_spec.nwalkers)
	#	if i % 
	#	p0[i] = np.random.normal(vp_params_flat[i],0.01,size=obs_spec.nwalkers)
 
	return np.transpose(p0)

def create_uniform_init(n_component,nwalkers):
	
	c = 299792.458
	logNs = np.random.uniform(0,10,    size=(n_component,nwalkers))
	bs = np.random.uniform(0,20, 	   size=(n_component,nwalkers))
	zs = np.random.uniform(-50/c,50/c, size=(n_component,nwalkers))

	alpha = np.array([logNs,bs,zs])
	p0 = alpha.swapaxes(0,1).reshape(n_component*3,nwalkers)
	return np.transpose(p0)

def run_kombine(output_fname):
	from likelihood import lnprob
	import kombine

	# define the MCMC parameters.
	ndim = 3*obs_spec.n_component
	p0 = create_guass_init(obs_spec.vp_params)

	# Start counting time
	t1 = time.time()

	# Set up the sampler
	print("Running MCMC...")
	sampler = kombine.Sampler(obs_spec.nwalkers, ndim, lnprob, processes=obs_spec.threads)

	# First do a rough burn in based on accetance rate.
	p_post_q = sampler.burnin(p0)
	p_post_q = sampler.run_mcmc(obs_spec.nsteps)
	
	np.save(output_fname,sampler.chain)
	t2 = time.time(); dt = (t2 - t1)/60.
	print("Done. dt = %.2f minutes" % (dt))

if __name__ == '__main__':
	
	run_kombine(obs_spec.chain_fname)
