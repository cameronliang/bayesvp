import numpy as np
import pylab as pl
import sys

from model import generic_prediction, ReadTransitionData
from observation import obs_spec 
from process_fits import read_mcmc_fits, write_mcmc_stats

def plot_spec():

    c = 299792.485
    burnin = 0.7 * obs_spec.nsteps

    mcmc_chain = np.load(obs_spec.chain_fname + '.npy')

    temp_flags = obs_spec.vp_params_flags[~np.isnan(obs_spec.vp_params_flags)]
    n_params = len(list(set(temp_flags)))
    samples = mcmc_chain[burnin:, :, :].reshape((-1, n_params))

    alpha = np.zeros(n_params)
    for n in xrange(n_params):
        alpha[n] = np.median(samples[:,n])

    model_flux = generic_prediction(alpha,obs_spec)
    # Use the first transition as the central wavelength
    # For now pick the 1st component, 1st wavelength region, 1st transiton's wavelength
    rest_wave = obs_spec.transitions_params_array[0][0][0][1]
    obs_spec_dv = c*(obs_spec.wave - rest_wave) / rest_wave

    summary = raw_input('Write best fit summary? (y/n): ')
    if summary == 'y':
        output_summary_fname = obs_spec.spec_path + '/vpfit_mcmc/bestfits_summary.dat' 
        write_mcmc_stats(obs_spec.chain_fname,output_summary_fname)

    plotting = raw_input('Plot model comparison? (y/n): ')
    if plotting == 'y':
        pl.rc('text', usetex=True)
        pl.step(obs_spec_dv,obs_spec.flux,'k',label=r'$\rm Data$')
        pl.step(obs_spec_dv,model_flux,'b',lw=2,label=r'$\rm Best\,Fit$')
        pl.step(obs_spec_dv,obs_spec.dflux,'r')
        pl.axhline(1,ls='--',c='g',lw=1.2)
        pl.axhline(0,ls='--',c='g',lw=1.2)
        pl.ylim([-0.1,1.4])
        dv = float(raw_input('Enter velocity range: '))
        pl.xlim([-dv,dv])
        pl.xlabel(r'$dv\,[\rm km/s]$')
        pl.ylabel(r'$\rm Normalized\,Flux$')
        pl.legend(loc=3)
        pl.show()
        #pl.savefig(obs_spec.spec_path + '/vpfit_mcmc/bestfit_spec.pdf',bbox_inches='tight',dpi=100)
        print('Written %svpfit_mcmc/bestfit_spec.pdf\n' % obs_spec.spec_path)
    
    output_model = raw_input('Write best fit model spectrum? (y/n): ')
    if output_model == 'y':
        np.savetxt(obs_spec.spec_path + '/vpfit_mcmc/bestfit_model.dat',
                np.c_[obs_spec.wave,obs_spec_dv,
                     obs_spec.flux, obs_spec.dflux,model_flux],
                header='wave\tdv\tflux\terror\tmodel')
        print('Written %svpfit_mcmc/bestfit_model.dat\n' % obs_spec.spec_path)
def main():
    plot_spec()

if __name__ == '__main__':
    sys.exit(int(main() or 0))