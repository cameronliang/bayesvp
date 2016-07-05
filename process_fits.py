import numpy as np


def compute_stats(x):
	xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
	xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
	xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)	
	return xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22
    
    
def write_mcmc_stats(mcmc_chain_fname,output_fname):

	chain = np.load(mcmc_chain_fname)
	
	burnin       = int(np.shape(chain)[0]*0.7)
	logN         = chain[burnin:,:,0].flatten()
	b_parameter  = chain[burnin:,:,1].flatten()
	redshift     = chain[burnin:,:,2].flatten()

	f = open(output_fname,'w')

	output_stats = compute_stats(logN)
	f.write('xmed\txm\txsd\txcfl11\txcfl12\t xcfl21\txcfl22\n')
	f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (output_stats[0],output_stats[1],
								  			  output_stats[2],output_stats[3],
								  			  output_stats[4],output_stats[5],
								  			  output_stats[6]))
	
	output_stats = compute_stats(b_parameter)
	f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (output_stats[0],output_stats[1],
								  			  output_stats[2],output_stats[3],
								  			  output_stats[4],output_stats[5],
								  			  output_stats[6]))

	output_stats = compute_stats(redshift)
	f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (output_stats[0],output_stats[1],
								  			  output_stats[2],output_stats[3],
								  			  output_stats[4],output_stats[5],
								  			  output_stats[6]))	
	f.close()
	print output_fname

	return 


def read_mcmc_fits(mcmc_chain_fname,para_name):
    
    my_dict = {'logN':0, 'b':1,'z':2}
    col_num = my_dict[para_name]
    chain = np.load(mcmc_chain_fname)

    burnin       = int(np.shape(chain)[0]*0.7)
    x            = chain[burnin:,:,col_num].flatten()
    xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22 = compute_stats(x)

    return xmed 