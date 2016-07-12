########################################################################################
#
#   		(c) Cameron J. Liang
#		    University of Chicago
#     		jwliang@oddjob.uchicago.edu
#       	BayesVP: A Full Bayesian Approach to Voigt Profile Fitting
########################################################################################


MCMC VPFIT assumes standard Python libraries: numpy, scipy, matplotlib and pyfits. 
I recommend installing the Enthought Canopy which comes with all the scientific 
libraries you need. You can find it here: https://www.enthought.com/products/canopy/

In addition, you will need an MCMC sampler. In Vogit profiles, usually we fit multiple components with many parameters, we will use the sampler KOMBINE developed by Ben Far from the University of Chicago and LIGO collaboration; the easist way is to install is using pip:

pip -U install KOMBINE

Another useful sampler is the Goodman-Weare Affine Parameter Sampler. The Implementation in python is developed by Dan Foreman-Mackey et al, along with a useful tool to plot the chains (triangle.py). 

pip -U install emcee
pip -U install triangle


------------------------------------------------------------------------------------------

Notes/Tips/Cautions:

1. For placing constraints for upper limit, one should not initialize walkers too far away from 'reasonable' parameters(e.g., column density). For example, if one knows logN= 15 is clearly too large given the data, then walkers should be initialized such that they do not waste time to get back to smaller logN and/or get stuck at larger logN. 

2. For upper limits, it is better to fix the redshift of the desire system in order to place a constraints. 

3. In some cases, the data are contaminated by some other lines, one can skip this contaminated region. 
	e.g., say, from (1215 1219) is the ideal region, but region from 1216 - 1217 is contaminated. Then just select regions in the config file, by breaking the wanted region into two regions (and so forth).
	1215 1216
	1217 1219

------------------------------------------------------------------------------------------


Example config file format:

\# test for saturated HI 					# Comment

! another type of comment 					# Comment 

/home/user_xxxx/tests 						# Path to spectrum

mcmc_chain.npy 								# Output fname

200 1000 16 								# nwalkers, nsteps, nthreads

%% uv_qso_5.spec 1546.8 1552.2 1333.5 1335  # spectrum_fname, [wave_start,wave_end], [..., ...], etc

% C IV 13.5 30 0.0001   					# Atom state logN b z

% C IV 14 10 0.0007

% C II 13 31 0.0001
