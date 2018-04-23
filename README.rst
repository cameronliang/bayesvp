
bayesvp
========

``bayesvp`` is a Bayesian MCMC parallel Voigt profile fitting routine. ``bayesvp`` provides a number of helpful executable scripts that work with command line arguments (saved in your environment ``PATH``). The main functionality is the MCMC Voigt profile fitting (``bvpfit``) where the user supplies a config file that specifies parameters for the fitting. These include parameter priors, number of walkers, parallel threads, line spread function, continuum model, Bayesian model comparisons, and etc. There are utility functions that allow users to quickly create an example config file, process and plot the chains, process and plot the best fit models and more. You can find more details on the code paper, [Liang & Kravtsov 2017](http://adsabs.harvard.edu/abs/2017arXiv171009852L) or a related paper [Liang et al. 2017](http://adsabs.harvard.edu/abs/2017arXiv171000411L)


Installation
------------

I recommend installing bayesvp with pip with the ``--user`` flag: 

    pip install bayesvp --user

This usually puts the executable scripts in ``~/.local/bin``. Make sure that this is in your PATH. 

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


Usage and Tests:
-------------------

You can run a full test example by executing: 

    bvpfit --test -pc

If the optional ``-pc`` flag is supplied, the default config file and a log are written to the current directory at which the command is run. 

This will run an MCMC fit with the default config file and test spectrum (./data/example). 
After the fit is finished, to process the MCMC chain, you can type: 

    bvp_process_model --test


You can create your own default config file and modify it to suit the needs of your particular absorption line system. Use -a for the automatic flag. 

    bvp_write_config -a 

These executables accept command line arguments. For example, to get more info on the 
usage of bvpfit, simply type: 

    bvpfit -h

You may want to use the newly generated default config file after the test to set up absorption line systems of your own. Instead of ``--test``, you can supply your own config 
file.

    bvpfit full_path_to_my_own_config_file.dat

It should just be this easy if ``bayesvp`` is installed correctly and your environment ``PATH`` knows the location of these executables. 

Required libraries:
-------------------

1) numpy, scipy, matplotlib and pyfits. 

2) MCMC Samplers ([kombine](http://home.uchicago.edu/~farr/kombine/kombine.html) and/or [emcee](http://dan.iel.fm/emcee/current/))

Notes/Tips/Cautions:
--------------------

1. For placing constraints on non-detections (i.e., upper limits), one should not initialize walkers too far away from 'reasonable' parameters(e.g., column density or redshift if you know it from somewhere else). For example, if one knows logN= 15 is clearly too large given the data, then walkers should be initialized such that they do not waste time to get back to smaller logN and/or get stuck at larger logN. 

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

If you use ``bayesvp``, please cite the paper: 

    @ARTICLE{Liang2017,
       author = {{Liang}, C. and {Kravtsov}, A. and {Agertz}, O.},
        title = "{Observing the circumgalactic medium of simulated galaxies through synthetic absorption spectra}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1710.00411},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2017,
        month = oct,
       adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171000411L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


or you can cite this one: 

    @ARTICLE{LiangKravtsov2017,
       author = {{Liang}, C. and {Kravtsov}, A.},
        title = "{BayesVP: a Bayesian Voigt profile fitting package}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1710.09852},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2017,
        month = oct,
       adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171009852L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

