import numpy as np
import pylab as pl
import sys

from model import model_prediction, ReadTransitionData
from observation import obs_spec 
from process_fits import read_mcmc_fits   

def plot_spec():

    transition_data = ReadTransitionData()
    c = 299792.485

    mcmc_chain_fname = obs_spec.chain_fname
    best_logN = read_mcmc_fits(mcmc_chain_fname,'logN')
    best_b = read_mcmc_fits(mcmc_chain_fname,'b')
    best_z = read_mcmc_fits(mcmc_chain_fname,'z')
    
    alpha = np.array([best_logN,best_b,best_z])
    print('logN = %.2f' % best_logN)
    print('b    = %.2f' % best_b)
    print('z    = %.5f' % best_z)
        
    model_flux = model_prediction(alpha,obs_spec.wave,obs_spec.n_component,obs_spec.transition_names)
    rest_wave = transition_data[obs_spec.transition_names[0]].wave
    obs_spec_dv = c*(obs_spec.wave - rest_wave) / rest_wave 
    pl.step(obs_spec_dv,obs_spec.flux,'k',label=r'$\rm data$')
    pl.step(obs_spec_dv,model_flux,'b',lw=1.5,label=r'$\rm best\,fit$')
    pl.step(obs_spec_dv,obs_spec.dflux,'r')
    
    pl.ylim([0,1.4])
    #pl.xlim([-1000,1000])
    pl.xlabel(r'$dv[km/s]$')
    pl.ylabel(r'$\rm Flux$')
    pl.legend(loc='best')
    pl.show()
    #pl.savefig(obs_spec.output_fname[:-4] + '_bestfit_spec_snr10.png',bbox_inches='tight',dpi=100)
    
    
def main():
    plot_spec()

if __name__ == '__main__':
    sys.exit(int(main() or 0))