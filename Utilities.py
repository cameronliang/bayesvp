import numpy as np
import re
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

###############################################################################
# Model Comparisons: Bayesian Evidence / BIC / AIC  
###############################################################################

def BIC_gaussian_kernel(chain_fname,data_length):
	"""
	Bayesian information criterion
	Only valid if data_length >> n_params

	# Note that bandwidth of kernel is set to 1 universally
	"""
	chain = np.load(chain_fname + '.npy')
	n_params = np.shape(chain)[-1]
	samples = np.transpose(chain.reshape((-1,n_params)))
	kde = KernelDensity(kernel='gaussian',bandwidth=1).fit(samples)

	# Best fit = medians of the distribution
	medians = np.median(samples,axis=0) # shape = (n_params,)
	medians = medians.reshape(1,-1) 	# Reshape to (1,n_params)
	
	log10_L = float(kde.score_samples(medians)) 
	lnL = log10_L /np.log10(2.7182818)
	 
	return -2*lnL + n_params*np.log(data_length)


def BIC_best_estimator(chain_fname,data_length):
	"""
	Bayesian information criterion
	Only valid if data_length >> n_params
	"""
	chain = np.load(chain_fname + '.npy')
	n_params = np.shape(chain)[-1]
	samples = np.transpose(chain.reshape((-1,n_params)))

	# Grid search cross-validation to optimize the bandwidth
	params = {'bandwidth': np.logspace(-2, 2, 20)}
	grid = GridSearchCV(KernelDensity(), params, cv=2)
	grid.fit(samples)

	# Use the best estimator determined by learning in sample
	kde = grid.best_estimator_

	# Best fit = medians of the distribution
	medians = np.median(samples,axis=0) # shape = (n_params,)
	medians = medians.reshape(1,-1) 	# Reshape to (1,n_params)

	log10_L = float(kde.score_samples(medians)) 
	lnL = log10_L /np.log10(2.7182818)

	return -2*lnL + n_params*np.log(data_length)

###############################################################################
# Process chain
###############################################################################
def compute_stats(x):
	xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
	xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
	xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)	
	return xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22
    

def read_mcmc_fits(mcmc_chain_fname,para_name):
    
	my_dict = {'logN':0, 'b':1,'z':2}
	col_num = my_dict[para_name]
	chain = np.load(mcmc_chain_fname)
	burnin_fraction = 0.5
	burnin = int(np.shape(chain)[0]*burnin_fraction)
	x = chain[burnin:,:,col_num].flatten()
	xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22 = compute_stats(x)

	return xmed 

def write_mcmc_stats(mcmc_chain_fname,output_fname):
	burnin_fraction = 0.5
	chain = np.load(mcmc_chain_fname)
	
	burnin       = int(np.shape(chain)[0]*burnin_fraction)

	f = open(output_fname,'w')
	f.write('x_med\tx_mean\tx_std\tx_cfl11\tx_cfl12\t x_cfl21\tx_cfl22\n')
	
	n_params = np.shape(chain)[-1]
	for i in xrange(n_params):
		x            = chain[burnin:,:,i].flatten()
		output_stats = compute_stats(x)
		f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % 
				(output_stats[0],output_stats[1],
				 output_stats[2],output_stats[3],
				 output_stats[4],output_stats[5],
				 output_stats[6]))
		
	f.close()
	print('Written %s: ' % output_fname)

	return


###############################################################################
# Line Spread Function 
###############################################################################

def gaussian_kernel(std):
    var = std**2
    size = 8*std +1 # this should gaurantee size to be odd.
    x = np.linspace(-100,100,size)
    norm = 1/(2*np.pi*var)
    return norm*np.exp(-(x**2/(2*std**2)))

def convolve_lsf(flux,lsf):
    # convolve 1-flux to remove edge effects wihtout using padding
    conv_flux = 1-np.convolve(1-flux,lsf,mode='same') /np.sum(lsf)
    return conv_flux



###############################################################################
# Convergence 
###############################################################################


def gr_indicator():
	"""
	Gelman-Rubin Indicator 
	"""
	return 0




###############################################################################
# Others
###############################################################################

def printline():
	print("-------------------------------------------------------------------")

def determine_autovp(config_fname):
    
	# Read and filter empty lines
	all_lines = filter(None,(line.rstrip() for line in open(config_fname)))

	normal_lines = []   # all lines except line with '!'
	component_line = [] # lines with one '%'
	auto_vp = False; n_component_max = 0
	for line in all_lines:
		if line.startswith('!'): 
			if re.search('auto', line) or re.search('AUTO', line): 
				line = line.split(' ')
				n_component_max = int(line[2])
				auto_vp = True

		elif re.search('%',line):
			if line.split(' ')[0] == '%':
				component_line.append(line)
			else:
				normal_lines.append(line)
		else:
			normal_lines.append(line)

	def replicate_config(config_fname,normal_lines, 
						component_line,n_component):
		
		# Find ind of the extention
		dot_index =  config_fname.find('.')
		new_config_fname = (config_fname[:dot_index] + str(n_component)
							 + config_fname[dot_index:])
		f = open(new_config_fname,'w')
		for line in normal_lines:
			f.write(line); f.write('\n')
		for line in component_line:
			for n in xrange(1,n_component+1):
				f.write(line); f.write('\n')
		f.close()

	for n in xrange(1,n_component_max+1):
		replicate_config(config_fname,normal_lines,component_line,n)
	return auto_vp, n_component_max

def print_config_params(obs_spec):

    print('\n')
    print('Spectrum Path: %s'     % obs_spec.spec_path)
    print('Spectrum name: %s'     % obs_spec.spec_short_fname)
    print('Fitting %i components with transitions: ' % obs_spec.n_component)
    for i in xrange(len(obs_spec.transitions_params_array)):
        for j in xrange(len(obs_spec.transitions_params_array[i])):
            if not np.isnan(obs_spec.transitions_params_array[i][j]).any():
                for k in xrange(len(obs_spec.transitions_params_array[i][j])):
                    rest_wavelength = obs_spec.transitions_params_array[i][j][k][1] 
                    print('    Transitions Wavelength: %.3f' % rest_wavelength)

    print('Selected data wavelegnth region:')
    for i in xrange(len(obs_spec.wave_begins)):
        print('    (%.3f, %.3f)' % (obs_spec.wave_begins[i],obs_spec.wave_ends[i])) 
    print('\n')

