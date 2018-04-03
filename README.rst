
bayesvp
========

``bayesvp`` is a Bayesian MCMC Voigt profile fitting routine. ``bayesvp`` provides a number of helpful executable scripts that work with command line arguments (saved in your environment ``PATH``). The main functionality is, of course, the MCMC Voigt profile fitting (``bvpfit``), that user supplies a config file that specifies parameter priors, number of walkers, parallel threads, and etc. There are utility functions that allow users to quickly create an example config file, plot the chain, plot and process the best fit models.  


Installation
------------

I recommend installing bayesvp with pip with the ``--user`` flag: 

    pip install bayesvp --user

You can also install it system-wide and might need to add ``sudo`` in the beginning. 

After installing ``bayesvp``, you should run its unit tests to ensure the package works as expected. The simplest way to do this is inside a python shell: 

    from bayesvp.tests import run_tests

The output should look something like this: 

    test_config_file_exists (bayesvp.tests.test_config.TCConfigFile) ... ok
    test_default_no_continuum_params (bayesvp.tests.test_config.TCConfigFile) ... ok
    test_example_mcmc_params (bayesvp.tests.test_config.TCConfigFile) ... ok
    ...
    test_prior (bayesvp.tests.test_likelihood.TCPosterior) ... ok
    test_general_intensity (bayesvp.tests.test_model.TCSingleVP) ... ok
    test_simple_spec (bayesvp.tests.test_model.TCSingleVP) ... ok

    ----------------------------------------------------------------------
    Ran 13 tests in 3.654s

    OK

If you encounter any error, please send output to the author. 

You can also run a full test example by executing: 

    bvpfit --test

This will run an MCMC fit with the default config file and test spectrum (./data/example). 
After the fit is finished, to process the MCMC chain, you can type: 

    bvp_process_model --test


You can create your own default config file and modify it to suit the needs of your particular absorption line system. Use -a for the automatic flag. 

    bvp_write_config -a 

These executables accept command line arguments. For example, to get more info on the 
usage of bvpfit, simply type: 

    bvpfit -h


Required libraries:
--------------------

1) numpy, scipy, matplotlib and pyfits. 

2) sklearn

3) MCMC Sampler ([kombine](http://home.uchicago.edu/~farr/kombine/kombine.html) and/or [emcee](http://dan.iel.fm/emcee/current/))

4) [Corner plot](https://corner.readthedocs.io/en/latest/)

Notes/Tips/Cautions:
--------------------

1. For placing constraints for upper limits, one should not initialize walkers too far away from 'reasonable' parameters(e.g., column density or redshift if you know it from somewhere else). For example, if one knows logN= 15 is clearly too large given the data, then walkers should be initialized such that they do not waste time to get back to smaller logN and/or get stuck at larger logN. 

2. For upper limits, it is better to fix the redshift of the desired system in order to place constraints. 

3. In some cases, the data are contaminated by some other lines, one can skip this contaminated region. 
    e.g., say, from (1215 1219) is the ideal region, but region from 1216 - 1217 is contaminated. Then just select regions in the config file, by breaking the wanted region into two regions (and so forth).
    1215 1216
    1217 1219

4. One can add a continuum model (polynomial of degree n) by adding a new line: “continuum 1”, which will add a linear continuum with two extra of parameters (offset and a slope). We do not recommend to go higher than degree 2. The continuum model is restricted to fitting only one segment of the spectrum. Simultaneous fitting with multiple lines is not currently supported.


License & Citing
----------------

Author:        Cameron Liang (jwliang@oddjob.uchicago.edu; cameron.liang@gmail.com)

Contributors:  Andrey Kravtsov

License:       MIT. Copyright (c) 2017-2018

