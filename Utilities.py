import numpy as np

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
# Line Spread Function Related
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
# Convergence related
###############################################################################


def gr_indicator():
	return 0

