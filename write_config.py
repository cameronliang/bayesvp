import numpy as np
import os

def interactive_var():
    spec_path    = raw_input('Path to spectrum:\n')
    spec_fname   = raw_input('Spectrum filename: ')
    output_chain_fname = raw_input('filename for output chain: ')
    atom         = raw_input('atom: ')
    state        = raw_input('state: ')
    auto         = int(raw_input('Maximum number of components to try: '))
    wave_start   = float(raw_input('Starting wavelength: '))
    wave_end     = float(raw_input('Ending wavelength: '))
    
    print('\nNow enter the priors. Press Enter for default values.')
    min_logN = 0; max_logN = 24 

    try:
        min_logN = float(raw_input('min logN = '))
    except ValueError:
        min_logN = 0

    try:
        max_logN = float(raw_input('max logN = '))
    except ValueError:
        max_logN = 24

    try:
        min_b = float(raw_input('min b = '))
    except ValueError:
        min_b = 0

    try:
        max_b = float(raw_input('max b = '))
    except ValueError:
        max_b = 200

    try:
        central_redshift = float(raw_input('central redshift = '))
    except ValueError:
        central_redshift = 0

    try:
        velocity_range   = float(raw_input('velocity range [km/s] = '))
    except ValueError:
        velocity_range = 500

    print('\nNow enter the MCMC parameters..')
    try:
        nwalkers = int(raw_input('Number of walkers: '))
    except ValueError:
        nwalkers = 200
    
    try:
        nsteps   = int(raw_input('Number of steps:  '))
    except ValueError:
        nsteps = 1000

    try:
        nthreads = int(raw_input('Number of processes: '))
    except ValueError:
        nthreads = 2

    model_selection = raw_input('Model selection method bic(default),aic,bf: ')
    if model_selection == '':
        model_selection = 'bic'
    
    mcmc_sampler = raw_input('MCMC sampler kombine(default), emcee: ')
    if mcmc_sampler == '':
        mcmc_sampler = 'kombine'



def write_single_config():

    interactive_var()

    config_path = spec_path + '/vp_configs'
    if not os.path.isdir(config_path):
        os.mkdir(config_path)

    config_fname = config_path + '/config_' + atom + state + '.dat'
    f = open(config_fname,'w')
    f.write('spec_path %s\n' % spec_path)
    if auto > 1:
        f.write('! auto %d\n' % auto)
    f.write('output %s\n' % output_chain_fname)
    f.write('mcmc %d %d %d %s %s\n' % (nwalkers,nsteps,nthreads,
                                model_selection,mcmc_sampler))
    f.write('%%%% %s %.6f %.6f\n' % (spec_fname,wave_start,wave_end))
    f.write('%% %s %s 15 30 %f\n' % (atom, state,central_redshift))

    f.write('logN %.2f %.2f\n' % (min_logN,    max_logN))
    f.write('b    %.2f %.2f\n' % (min_b,       max_b))
    f.write('z    %.6f %.2f\n' % (central_redshift,velocity_range))
    f.close()

if __name__ == '__main__':

    write_single_config()
