import numpy as np
import pylab as pl
import sys

from model import generic_prediction, ReadTransitionData
from observation import obs_spec 
from process_fits import read_mcmc_fits   

def plot_spec():

    c = 299792.485

    mcmc_chain_fname = obs_spec.chain_fname
    best_logN = read_mcmc_fits(mcmc_chain_fname,'logN')
    best_b = read_mcmc_fits(mcmc_chain_fname,'b')
    best_z = read_mcmc_fits(mcmc_chain_fname,'z')
    
    alpha = np.array([best_logN,best_b,best_z])
    print('logN = %.2f' % best_logN)
    print('b    = %.2f' % best_b)
    print('z    = %.5f' % best_z)

    model_flux = generic_prediction(alpha,obs_spec)

    # Use the first transition as the central wavelength
    rest_wave = obs_spec.transitions_params_array[0][0][1]
    obs_spec_dv = c*(obs_spec.wave - rest_wave) / rest_wave 
    pl.step(obs_spec_dv,obs_spec.flux,'k',label=r'$\rm data$')
    pl.step(obs_spec_dv,model_flux,'b',lw=1.5,label=r'$\rm best\,fit$')
    pl.step(obs_spec_dv,obs_spec.dflux,'r')
    
    pl.ylim([0,1.4])
    pl.xlabel(r'$dv[km/s]$')
    pl.ylabel(r'$\rm Flux$')
    pl.legend(loc='best')
    pl.savefig(obs_spec.spec_path + '/vpfit_mcmc/bestfit_spec.pdf',bbox_inches='tight',dpi=100)
    
    
def main():
    plot_spec()

if __name__ == '__main__':
    sys.exit(int(main() or 0))