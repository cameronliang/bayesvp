################################################################################
#
# utilities.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Utility functions used throughout BayesVP      
################################################################################

import numpy as np
import re
import os
import sys
import argparse
from scipy.special import gamma

import matplotlib.pyplot as plt



###############################################################################
# Model Comparisons: Bayesian Evidence / BIC / AIC  
###############################################################################

def determine_autovp(config_fname):
	"""
	Determine based on config file if automatic 
	mode is chosen for vpfit. If auto = True, then 
	reproduce the (n_max - n_min) number of files with 
	that number components
	"""
	def replicate_config(config_fname,normal_lines, 
						component_line,n_component):

		basename_with_path, config_extension = os.path.splitext(config_fname)
		new_config_fname = (basename_with_path + str(n_component)
							 + config_extension)

		f = open(new_config_fname,'w')
		for line in normal_lines:
			temp_line = line.split(' ')
			
			# Add the number of component at the end of output chain filename
			if 'output' in temp_line or 'chain' in temp_line:
				output_name = temp_line[1] + str(n_component)
				f.write(temp_line[0] + ' '); f.write(output_name)
				f.write('\n')
			else:
				f.write(line); f.write('\n')
		
		for line in component_line:
			for n in xrange(1,n_component+1):
				f.write(line); f.write('\n')
		f.close()

	# Read and filter empty lines
	all_lines = filter(None,(line.rstrip() for line in open(config_fname)))

	normal_lines = []   # all lines except line with '!'
	component_line = [] # lines with one '%'
	auto_vp = False; n_component_max = 1; n_component_min = 1
	for line in all_lines:
		if line.startswith('!'): 
			if re.search('auto', line) or re.search('AUTO', line): 
				line = line.split(' ')
				if len(line) == 3:
					n_component_min = 1 
					n_component_max = int(line[2])
				elif len(line) == 4: 
					n_component_min = int(line[2])
					n_component_max = int(line[3])
				auto_vp = True

		elif re.search('%',line):
			if line.split(' ')[0] == '%':
				component_line.append(line)
			else:
				normal_lines.append(line)
		else:
			normal_lines.append(line)

	if auto_vp:
		# Produce Config files
		for n in xrange(n_component_min,n_component_max+1):
			replicate_config(config_fname,normal_lines,component_line,n)

	return auto_vp, n_component_min, n_component_max


def model_info_criterion(obs_spec_obj):
	"""
	Use either aic,bic or bayes factor for model selection

	Aikake Information Criterion (AIC); 
	Bayesian Information Criterion (BIC); 
	see Eqn (4.17) and Eqn (5.35) Ivezic+ "Statistics, 
	Data Mining and Machine Learning in Astronomy" (2014)
	"""

	#if obs_spec_obj.model_selection.lower() in ('odds','bf'):
	#	return local_density_bf(obs_spec_obj) 

	from Likelihood import Posterior
	# Define the posterior function based on data
	lnprob = Posterior(obs_spec_obj)

	data_length = len(obs_spec_obj.flux)

	chain = np.load(obs_spec_obj.chain_fname + '.npy')
	n_params = np.shape(chain)[-1]
	samples = chain.reshape((-1,n_params))
	medians = np.median(samples,axis=0) # shape = (n_params,)

	log10_L = lnprob(medians)
	lnL = log10_L /np.log10(2.7182818)
	if obs_spec_obj.model_selection.lower() == 'aic':
		return -2*lnL + 2*n_params + 2*n_params*(n_params+1)/(np.log(data_length) - n_params - 1)
	elif obs_spec_obj.model_selection.lower() == 'bic':
		return -2*lnL + n_params*np.log(data_length)

	else:
		print('model_selection is not defined to either be aic or bic')
		sys.exit()

#def estimate_bayes_factor(traces, logp, r=0.05):
#	"""
#	Esitmate Odds ratios in a random subsample of the chains in MCMC
#	AstroML (see Eqn 5.127, pg 237)
#	"""
#	from sklearn.neighbors import BallTree
#	
#	ndim, nsteps = traces.shape # [ndim,number of steps in chain]
#
#	# compute volume of a n-dimensional (ndim) sphere of radius r
#	Vr = np.pi ** (0.5 * ndim) / gamma(0.5 * ndim + 1) * (r ** ndim)
#
#	# use neighbor count within r as a density estimator
#	bt = BallTree(traces.T)
#	count = bt.query_radius(traces.T, r=r, count_only=True)
#
#	# BF = N*p/rho
#	bf = logp + np.log(nsteps) + np.log(Vr) - np.log(count) #log10(bf)
#
#	p25, p50, p75 = np.percentile(bf, [25, 50, 75])
#	return p50, 0.7413 * (p75 - p25)

########################################################################

#def local_density_bf(obs_spec_obj):
#	"""
#	Bayes Factor: L(M) based on local density estimate
#	See (5.127) in Ivezic+ 2014 (AstroML)
#
#	Assuming we need only L(M) at a given point.
#	"""
#	from Likelihood import Posterior
#	from kombine.clustered_kde import ClusteredKDE
#
#	# Define the posterior function based on data
#	lnprob = Posterior(obs_spec_obj)
#	chain = np.load(obs_spec_obj.chain_fname + '.npy')
#	n_params = np.shape(chain)[-1]
#	samples = chain.reshape((-1,n_params))
#
#	# KDE estimate of the sample
#	sample_fraction = 0.2
#	n_sample = (obs_spec_obj.nsteps)*sample_fraction
#	ksample = ClusteredKDE(samples)
#	sub_sample = ksample.draw(n_sample) 
#
#	logp = np.zeros(n_sample)
#	for i in xrange(n_sample):
#		logp[i] = lnprob(sub_sample[i])
#
#	bf,dbf = estimate_bayes_factor(sub_sample.T,logp)
#	return bf

def compare_model(L1,L2,model_selection):
	"""
	Compare two Models L(M1) = L1 and L(M2) = L2. 
	if return True L1 wins; otherwise L2 wins. 

	For Bayes Factor/Odds ratio, it is the log10(L1/L2) being 
	compared. 
	"""

	if model_selection in ('bic', 'aic'):
		return L1 <= L2
	#elif model_selection in ('odds', 'BF','bf'):
	#	return L1-L2 >= 0

###############################################################################
# Others 
###############################################################################
#def bic_gaussian_kernel(chain_fname,data_length):
#	"""
#	Bayesian information criterion
#	Only valid if data_length >> n_params
#
#	# Note that bandwidth of kernel is set to 1 universally
#	"""
#	from sklearn.neighbors import KernelDensity
#
#	chain = np.load(chain_fname + '.npy')
#	n_params = np.shape(chain)[-1]
#	samples = chain.reshape((-1,n_params))
#	kde = KernelDensity(kernel='gaussian',bandwidth=1).fit(samples)
#
#	# Best fit = medians of the distribution
#	medians = np.median(samples,axis=0) # shape = (n_params,)
#	medians = medians.reshape(1,-1) 	# Reshape to (1,n_params)
#	
#	log10_L = float(kde.score_samples(medians)) 
#	lnL = log10_L /np.log10(2.7182818)
#	 
#	return -2*lnL + n_params*np.log(data_length)


###############################################################################
# Process chain
###############################################################################

def compute_stats(x):
	xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
	xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
	xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)	
	return xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22
    

def read_mcmc_fits(config_params_obj,para_name):
    
	my_dict = {'logN':0, 'b':1,'z':2}
	col_num = my_dict[para_name]
	chain = np.load(config_params_obj.chain_fname + '.npy')
	burnin = compute_burnin_GR(config_params_obj.chain_fname + '_GR.dat')
	x = chain[burnin:,:,col_num].flatten()
	xmed,xm,xsd,xcfl11, xcfl12, xcfl21,xcfl22 = compute_stats(x)
	return xmed 

def write_mcmc_stats(config_params_obj,output_fname):
	chain = np.load(config_params_obj.chain_fname + '.npy')
	burnin = compute_burnin_GR(config_params_obj.chain_fname + '_GR.dat')
	
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
	print('Written %s' % output_fname)

	return

def extrapolate_pdf(x,pdf,left_boundary_x,right_boundary_x,x_stepsize,slope=10):
    """ 
    Extrapolate the log10(pdf) outside the range of (min_x,max_x) with 
    some logarithmic slope 
    """
    np.seterr(all='ignore') # Ignore floating point warnings.
    log_pdf = np.log10(pdf)
    min_x = min(x); max_x = max(x)
    #x_stepsize = np.median(x[1:]-x[:-1])

    entered_left_condition = False
    if min_x >= left_boundary_x:

        # equation of a line with +10 slope going down to the left.
        left_added_x = np.arange(left_boundary_x,min_x,x_stepsize) 
        m = slope; b = log_pdf[0] - m*min_x
        left_pdf = m*left_added_x + b 
    
        # Combine the two segments    
        new_x = np.concatenate((left_added_x,x))
        log_pdf = np.concatenate((left_pdf,log_pdf))
        
        entered_left_condition = True
    
    if max_x <= right_boundary_x:
        
        # Equation of a line with -10 slope going down to the right.
        right_added_x = np.arange(max_x,right_boundary_x,x_stepsize)
        m = -slope; b = log_pdf[-1] - m*max_x
        right_pdf = m*right_added_x + b
        
        # In case new_x is not defined yet if not entered previous condition
        if entered_left_condition:
            new_x = np.concatenate((new_x,right_added_x))
        else:
            new_x = np.concatenate((x,right_added_x))
        log_pdf = np.concatenate((log_pdf,right_pdf))        

    # Normalize the pdf
	pdf_tmp  = 10**log_pdf/np.sum((10**log_pdf)*(x_stepsize))
	inds = np.where(pdf_tmp<0)[0]; pdf_tmp[inds] = np.min(pdf_tmp)
    log_pdf = np.log10(pdf_tmp)
    return new_x, log_pdf



###############################################################################
# Line Spread Function 
###############################################################################

def gaussian_kernel(std):
    var = std**2
    size = 8*std +1 # this gaurantees size to be odd.
    x = np.linspace(-100,100,size)
    norm = 1/(2*np.pi*var)
    return norm*np.exp(-(x**2/(2*std**2)))

def convolve_lsf(flux,lsf):
	if len(flux) < len(np.atleast_1d(lsf)):
		# Add padding to make sure to return the same length in flux.
		padding = np.ones(len(lsf)-len(flux)+1)
		flux = np.hstack([padding,flux])
    	
		conv_flux = 1-np.convolve(1-flux,lsf,mode='same') /np.sum(lsf)
		return conv_flux[len(padding):]

	else:
		# convolve 1-flux to remove edge effects wihtout using padding
		return 1-np.convolve(1-flux,lsf,mode='same') /np.sum(lsf)

###############################################################################
# Convergence 
###############################################################################

def gr_indicator(chain):
	"""
	Gelman-Rubin Indicator

	Parameters:
	-----------
	chain: array_like
		Multi-dimensional array of the chain with shape (nsteps,nwalkers,ndim)

	Returns
	-----------
	Rgrs: array_like
		Gelman-Rubin indicator with length of (ndim)
	"""
	nsteps,nwalkers,ndim = np.shape(chain)
	nsteps = float(nsteps); nwalkers = float(nwalkers)
    
	Rgrs = np.zeros(ndim)
	for n in xrange(ndim):
		x = chain[:,:,n]
		# average of within-chain variance over all walkers
		W = np.mean(np.var(x,axis=0)) # i.e within-chain variance
		mean_x_per_chain = np.mean(x,axis=0)
		mean_x = np.mean(mean_x_per_chain) 

		# Variance between chains 
		B = nsteps*np.sum((mean_x_per_chain - mean_x)**2) / (nwalkers-1)
		var_per_W = 1 - 1./nsteps + B/(W*nsteps) 

		Rgrs[n] = ((nwalkers+1)/nwalkers) * var_per_W - (nsteps-1)/(nwalkers*nsteps)
        
	return Rgrs 

def compute_burnin_GR(gr_fname,gr_threshold=1.005):
	"""
	Calculate the steps where the chains are 
	converged given a Gelman-Rubin (GR) threshod. 

	Parameters
	----------
	gr_fname:str
		Full path to the GR file with first column as steps
		and the rest as the values of GR for each model parameter
	gr_threshold:float
		The threshold for chains to be considered as converged.
		Default value = 1.01
	Returns
	----------
	burnin_steps: int
		The step number where the least converged parameter has 
		converged; (maxiumn of the steps of all parameters) 
	"""
	data = np.loadtxt(gr_fname,unpack=True)
	steps = data[0]; grs = data[1:]
	indices = np.argmax(grs<=gr_threshold,axis=1)
	# burnin_steps
	return int(np.max(steps[indices]))


###############################################################################
# Others
###############################################################################

def straight_line(x,m,b):
	return (m*x + b) #/ np.mean((m*x + b))

def linear_continuum(wave,flux,m,b):
	continuum = straight_line(wave,m,b)
	return flux * continuum

def get_transitions_params(atom,state,wave_start,wave_end,redshift):
	"""
	Extract the ionic and tranisiton properties based on the atom and 
	the state, given the redshift and observed wavelength regime.

	Parameters:
	----------
	atom: str
		names of atoms of interests (e.g., 'H', 'C','Si')
	state: str
		ionization state of the atom in roman numerals (e.g., 'I', 'II')
	wave_start: float
		the minimum observed wavelength 
	wave_end: float
		the maximum observed wavelength
	redshift: float
		redshift of the absorption line
	
	Return:
	----------
	transitions_params_array: array_like
		[oscilator strength, rest wavelength, dammping coefficient, mass of the atom] for all transitions within the wavelength interval. shape = (number of transitions, 4) 
	"""
	amu = 1.66053892e-24   # 1 atomic mass in grams

	# Absolute path for BayseVP
	data_path = os.path.dirname(os.path.abspath(__file__)) 
	data_file = data_path + '/data/atom.dat'
	atoms,states  = np.loadtxt(data_file, dtype=str,unpack=True,usecols=[0,1])
	wave,osc_f,Gamma,mass = np.loadtxt(data_file,unpack=True,usecols=[2,3,4,5])
	mass = mass*amu 

	inds = np.where((atoms == atom) & 
					(states == state) & 
					(wave >= wave_start/(1+redshift)) & 
					(wave < wave_end/(1+redshift)))[0]
 
	if len(inds) == 0:
		return np.empty(4)*np.nan
	else:
		return np.array([osc_f[inds],wave[inds],Gamma[inds], mass[inds]]).T


def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def triage(par, weights, parnames, nbins = 30, hist2d_color=plt.cm.PuBu,
		hist1d_color='#3681f9',figsize=[6,6], figname=None, fontsize=10,labelsize=10):
	"""
	Plot the multi-dimensional and marginalized posterior distribution (e.g., `corner` plot)

	Parameters:
	----------
	par: array
		sampled mcmc chains with shape (n_params,nsteps)
	weights: array
		wights of the chains (nominally=1 if all chains carry equal weights), same shape as par
	parnames: array
		parameter names 
	nbins: int
		number of bins in the histograms of PDF
	hist2d_color: str
		matplotlib colormap of the 2D histogram
	hist1d_color: str
		color for the 1D marginalized histogram
	figsize: list
		size of the figure. example: [6,6]
	figname: str
		full path and name of the figure to be written 
	fontsize: int
		fontsize of the labels 
	"""

	import matplotlib.gridspec as gridspec

	import itertools as it
	import scipy.optimize as opt
	from matplotlib.colors import LogNorm
	from matplotlib.ticker import MaxNLocator, NullLocator


	import warnings
	# ignore warnings if matplotlib version is older than 1.5.3
	warnings.simplefilter(action='ignore', category=FutureWarning) 
	
	# ignore warning for log10 of parameters with values <=0.
	warnings.simplefilter(action='ignore', category=UserWarning)
	

	npar = np.size(par[1,:])
	fig = plt.figure(figsize=figsize)
	gs = gridspec.GridSpec(npar, npar,wspace=0.05,hspace=0.05,
		width_ratios=[1,1,1],height_ratios=[1,1,1])
	
	for h,v in it.product(range(npar), range(npar)) :
		ax = plt.subplot(gs[h, v])


		x_min, x_max = np.min(par[:,v]), np.max(par[:,v])
		y_min, y_max = np.min(par[:,h]), np.max(par[:,h])

		ax.tick_params(axis='both', which='major', labelsize=labelsize)
		ax.tick_params(axis='both', which='minor', labelsize=labelsize)	
		if h < npar-1:
			ax.get_xaxis().set_ticklabels([])
		if v > 0:
			ax.get_yaxis().set_ticklabels([])

		if h > v :
			hvals, xedges, yedges = np.histogram2d(par[:,v], par[:,h], 
									weights=weights[:,0], bins = nbins)
			hvals = np.rot90(hvals)
			hvals = np.flipud(hvals)
				
			Hmasked = np.ma.masked_where(hvals==0, hvals)
			hvals = hvals / np.sum(hvals)        
				
			X,Y = np.meshgrid(xedges,yedges) 
				
			sig1 = opt.brentq( conf_interval, 0., 1., args=(hvals,0.683) )
			sig2 = opt.brentq( conf_interval, 0., 1., args=(hvals,0.953) )
			sig3 = opt.brentq( conf_interval, 0., 1., args=(hvals,0.997) )
			lvls = [sig3, sig2, sig1]   
						
			ax.pcolor(X, Y, (Hmasked), cmap=hist2d_color, norm = LogNorm())
			ax.contour(hvals, linewidths=(1.0, 0.5, 0.25), colors='lavender', 
							levels = lvls, norm = LogNorm(), extent = [xedges[0], 
							xedges[-1], yedges[0], yedges[-1]])

			ax.set_xlim([x_min, x_max])
			ax.set_ylim([y_min, y_max])

		elif v == h :
			ax.hist(par[:,h],bins = nbins,color=hist1d_color,histtype='step',lw=1.5)
		
			ax.yaxis.set_ticklabels([])

			hmedian = np.percentile(par[:,h],50)
			h16 = np.percentile(par[:,h],16)
			h84 = np.percentile(par[:,h],84)

			ax.axvline(hmedian,lw=0.8,ls='--',color='k')
			ax.axvline(h16,lw=0.8,ls='--',color='k')
			ax.axvline(h84,lw=0.8,ls='--',color='k')
			ax.set_xlim([x_min, x_max])

			ax.set_title(r'$%.2f^{+%.2f}_{-%.2f}$' % (hmedian,h84-hmedian,hmedian-h16),
						fontsize=fontsize)

		else :
			ax.axis('off')

		if v == 0:
			ax.set_ylabel(parnames[h],fontsize=fontsize)
			ax.get_yaxis().set_label_coords(-0.2,0.5)
			
		if h == npar-1:
			ax.set_xlabel(parnames[v],fontsize=fontsize)
			ax.get_xaxis().set_label_coords(0.5,-0.35)
			labels = ax.get_xticklabels()
			for label in labels: 
				label.set_rotation(80) 

	fig.get_tight_layout()
	if figname:
		plt.savefig(figname, dpi=120, bbox_inches='tight')


def printline():
	print("---------------------------------------------------------------------")


def get_bayesvp_Dir():
	return os.path.dirname(os.path.realpath(__file__))


class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def print_config_help():
    
	printline()
	printline()
	print('Example of config file content:')
	printline()
	print('spec_path test_path_to_spec')
	print('output o6')
	print('mcmc 100 200 4 bic kombine')
	print('%% OVI.spec 1030.000000 1033.000000')
	print('% O VI 15 30 0.000000')
	print('logN 10.00 18.00')
	print('b    0.00 100.00')
	print('z    0.000000 300.00')
	printline()
	printline()
	print('\n')

	printline()
	printline()
	print('Structure of a line in config file: ')
	print('parameter_type parameter(s)')
	printline()
	print('spec_path full_path_to_spectrum')
	print('output mcmc_chain_filename')
	print('mcmc walkers steps_per_walker parallel_threads MCMC_evidence_criterion MCMC_sampler')
	print('%% spectrum_file_name wave_begin1 wave_end1 wave_begin2 wave_end2 ...')
	print('% Atom State logN_guess b_guess z_guess')
	print('logN min_logN max_logN')
	print('b    min_b max_b')
	print('z    center_z dv_range')
	printline()
	printline()

	print('\n')

	printline()
	printline()
	print('optional lines/commands:')

	printline()
	print('lsf lsf_file_name\n')
	print('where lsf = line spread function to be convolved with model')
	print('lsf_file_name is assumed to be in spec_path/database')

	printline()
	print('continuum n \n')
	print('where n = polynomial degree; number of parameter = n+1')




	printline()
	print('! auto n\n')
	print('Automatically try up to n number of Voigt profile component')

	printline()
	print('# lines begin with # is an optional comment')
	printline()
