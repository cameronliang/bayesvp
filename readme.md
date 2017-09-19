########################################################################################
###
###   		(c) Cameron J. Liang
###		    The University of Chicago
###     	    jwliang@oddjob.uchicago.edu
###       	BayesVP: A Full Bayesian Approach to Voigt Profile Fitting
########################################################################################

------------------------------------------------------------------------------------------

BayseVP requires the following libraries:

1) numpy, scipy, matplotlib and pyfits. 

2) sklearn

3) MCMC Sampler

4) Corner plot

Python distributions:

[Anaconda](https://www.continuum.io/downloads)

[Enthought Canopy](https://www.enthought.com/products/canopy/)

In Vogit profiles fitting, we often need to fit multiple components with many parameters, we will use the sampler [KOMBINE](http://home.uchicago.edu/~farr/kombine/kombine.html) developed by Ben Far and Will Far from the University of Chicago and LIGO collaboration; the easist way is to install is using pip:

pip install kombine --user

Another useful sampler is the Goodman-Weare Affine Parameter Sampler [emcee](http://dan.iel.fm/emcee/current/). The Implementation in python is developed by Dan Foreman-Mackey et al, along with a useful tool to plot the chains (corner.py). 

pip install emcee --user

pip install corner --user

------------------------------------------------------------------------------------------

Notes/Tips/Cautions:

1. For placing constraints for upper limit, one should not initialize walkers too far away from 'reasonable' parameters(e.g., column density or redsfhit if you know it from somewhere else). For example, if one knows logN= 15 is clearly too large given the data, then walkers should be initialized such that they do not waste time to get back to smaller logN and/or get stuck at larger logN. 

2. For upper limits, it is better to fix the redshift of the desire system in order to place a constraints. 

3. In some cases, the data are contaminated by some other lines, one can skip this contaminated region. 
	e.g., say, from (1215 1219) is the ideal region, but region from 1216 - 1217 is contaminated. Then just select regions in the config file, by breaking the wanted region into two regions (and so forth).
	1215 1216
	1217 1219

4. One can add a continuum model (polynomial of degree n) by adding a new line: “continuum 1”, which will add a linear continuum with two extra of parameters (offset and a slope). We do not recommend to go higher than degree 2. The continuum model is restricted to fitting only one segment of the spectrum. Simultaneous fitting with multiple lines is not currently supported.

------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------

Features to be added/planned: 
* Thermally link b parameters
