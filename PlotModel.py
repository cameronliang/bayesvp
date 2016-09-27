import numpy as np
import matplotlib.pyplot as pl
import sys

from Model import generic_prediction
from Utilities import write_mcmc_stats, compute_burin_GR

def plot_model_comparison(config_params_obj,model_flux,rest_wave,redshift,dv):
    c = 299792.485 # [km/s]

    if rest_wave == None:
        # Use the first transition as the central wavelength
        rest_wave = config_params_obj.transitions_params_array[0][0][0][1]
    else:
        rest_wave = float(rest_wave)

    obs_spec_wave = config_params_obj.wave / (1+redshift) 
    obs_spec_dv = c*(obs_spec_wave - rest_wave) / rest_wave
    pl.rc('text', usetex=True)

    pl.figure(1)
    pl.step(obs_spec_dv,config_params_obj.flux,'k',label=r'$\rm Data$')
    pl.step(obs_spec_dv,model_flux,'b',lw=2,label=r'$\rm Best\,Fit$')
    pl.step(obs_spec_dv,config_params_obj.dflux,'r')
    pl.axhline(1,ls='--',c='g',lw=1.2)
    pl.axhline(0,ls='--',c='g',lw=1.2)
    pl.ylim([-0.1,1.4])
    pl.xlim([-dv,dv])
    pl.xlabel(r'$dv\,[\rm km/s]$')
    pl.ylabel(r'$\rm Normalized\,Flux$')
    pl.legend(loc=3)
    
    pl.savefig(config_params_obj.processed_product_path + '/modelspec_' + config_params_obj.chain_short_fname + '.pdf',bbox_inches='tight',dpi=100)
    pl.clf()

    print('Written %s/processed_products/best_fit_%s.pdf\n' % (config_params_obj.spec_path,config_params_obj.chain_short_fname))

def plot_gr_indicator(config_params_obj):
    gr_fname = config_params_obj.chain_fname + '_GR.dat'
    data = np.loadtxt(gr_fname,unpack=True)
    steps = data[0]; grs = data[1:]
    
    pl.figure(1,figsize=(6,6))
    for i in xrange(len(grs)):
        pl.plot(steps,grs[i],label=str(i))
    pl.legend(loc='best')
    pl.xscale('log')

    pl.xlabel(r'$N(\rm{steps})$')
    pl.ylabel(r'$R$')
    pl.savefig(config_params_obj.processed_product_path + '/' + config_params_obj.chain_short_fname + '_GR.pdf',bbox_inches='tight',dpi=100)

def write_model_spectrum(config_params_obj,model_flux):
    np.savetxt(config_params_obj.processed_product_path + 
            '/spec_' + config_params_obj.chain_short_fname+'.dat',
            np.c_[config_params_obj.wave,config_params_obj.flux, 
            config_params_obj.dflux,model_flux],
            header='wave\tflux\terror\tmodel')
    print('Written %s/processed_products/modelspec.dat' % config_params_obj.spec_path)
    
def write_model_summary(config_params_obj):
    mcmc_chain_fname = config_params_obj.chain_fname + '.npy'
    output_summary_fname = config_params_obj.processed_product_path + '/params_'+config_params_obj.chain_short_fname+'.dat'
    write_mcmc_stats(config_params_obj,output_summary_fname)



def process_model(obs_spec,redshift,dv):
    """Quick comparison and diagnostic of the best fit model"""
    
    mcmc_chain_fname = obs_spec.chain_fname + '.npy'
    mcmc_chain = np.load(mcmc_chain_fname)

    # Obtain best fit parameters and model flux (medians of the chains)
    temp_flags = obs_spec.vp_params_flags[~np.isnan(obs_spec.vp_params_flags)]
    n_params = len(list(set(temp_flags)))

    burnin = compute_burin_GR(obs_spec.chain_fname + '_GR.dat')
    samples = mcmc_chain[burnin:, :, :].reshape((-1, n_params))
    alpha = np.median(samples,axis=0)
    
    model_flux = generic_prediction(alpha,obs_spec)

    # Write best fit parameters summary file
    write_model_summary(obs_spec)

    # Write to file for original fitted data and best-fit model flux    
    write_model_spectrum(obs_spec,model_flux)

    # Plot the best fit for visual comparison
    rest_wave = obs_spec.transitions_params_array[0][0][0][1]
    plot_model_comparison(obs_spec,model_flux,rest_wave,redshift,dv)

    # plot the evolution of GR indicator as a function of steps
    plot_gr_indicator(obs_spec)

def main(config_fname,redshift,dv):
    from Config import DefineParams
    obs_spec = DefineParams(config_fname)

    redshift = float(redshift)
    dv       = float(dv)
    process_model(obs_spec,redshift,dv)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('python full_path_to_config')
    elif len(sys.argv) == 2:
        config_fname = sys.argv[1]
        print('python full_path_to_config redshift dv_range_for_plotting')
        redshift = float(raw_input('Redshift of system = '))
        dv       = float(raw_input('velocity range of spectrum plot = '))
        main(config_fname,redshift,dv)
    elif len(sys.argv) == 4:
        config_fname = sys.argv[1]
        redshift     = sys.argv[2] 
        dv           = sys.argv[3]
        main(config_fname,redshift,dv)
    else:
        print('python full_path_to_config redshift dv_range_for_plotting')
        exit()