import numpy as np
import matplotlib.pyplot as pl
import corner

from Model import generic_prediction
from Utilities import write_mcmc_stats, compute_burin_GR
from Config import DefineParams

class ProcessModel:

    def __init__(self,config_param):

        self.config_param = config_param
        mcmc_chain_fname = config_param.chain_fname + '.npy'
        mcmc_chain = np.load(mcmc_chain_fname)

        # Obtain best fit parameters and model flux (medians of the chains)
        temp_flags = config_param.vp_params_flags[~np.isnan(config_param.vp_params_flags)]
        self.n_params = len(list(set(temp_flags)))

        # MCMC chains
        self.burnin = compute_burin_GR(config_param.chain_fname + '_GR.dat')
        self.samples = mcmc_chain[self.burnin:, :, :].reshape((-1, self.n_params))
        
        # Best fit model parameters and spectrum flux
        self.alpha = np.median(self.samples,axis=0)
        self.model_flux = generic_prediction(self.alpha,config_param)

    def plot_model_comparison(self,redshift,dv,central_wave=None):
        """
        Plot best fit model onto spectrum for visual inspection 
        """
        c = 299792.485 # [km/s]

        if central_wave == None:
            # Use the first transition as the central wavelength
            central_wave = self.config_param.transitions_params_array[0][0][0][1]
        else:
            central_wave = float(central_wave)

        obs_spec_wave = self.config_param.wave / (1+redshift) 
        obs_spec_dv = c*(obs_spec_wave - central_wave) / central_wave
        pl.rc('text', usetex=True)

        pl.figure(1)
        pl.step(obs_spec_dv,self.config_param.flux,'k',label=r'$\rm Data$')
        pl.step(obs_spec_dv,self.model_flux,'b',lw=2,label=r'$\rm Best\,Fit$')
        pl.step(obs_spec_dv,self.config_param.dflux,'r')
        pl.axhline(1,ls='--',c='g',lw=1.2)
        pl.axhline(0,ls='--',c='g',lw=1.2)
        pl.ylim([-0.1,1.4])
        pl.xlim([-dv,dv])
        pl.xlabel(r'$dv\,[\rm km/s]$')
        pl.ylabel(r'$\rm Normalized\,Flux$')
        pl.legend(loc=3)
        
        pl.savefig(self.config_param.processed_product_path + '/modelspec_' + self.config_param.chain_short_fname + '.pdf',bbox_inches='tight',dpi=100)
        pl.clf()

        print('Written %s/processed_products/best_fit_%s.pdf\n' % (self.config_param.spec_path,self.config_param.chain_short_fname))

    def plot_gr_indicator(self):
        """
        Make plot for the evolution of GR indicator 
        as a function of steps
        """
        gr_fname = self.config_param.chain_fname + '_GR.dat'
        data = np.loadtxt(gr_fname,unpack=True)
        steps = data[0]; grs = data[1:]
        
        pl.figure(1,figsize=(6,6))
        for i in xrange(len(grs)):
            pl.plot(steps,grs[i],label=str(i))
        pl.legend(loc='best')
        pl.xscale('log')

        pl.xlabel(r'$N(\rm{steps})$')
        pl.ylabel(r'$R_{\rm GR}$')
        pl.savefig(self.config_param.processed_product_path + '/GR_' + self.config_param.chain_short_fname + '.pdf',bbox_inches='tight',dpi=100)
        pl.clf()

    def corner_plot(self):
        """
        Make triangle plot for visuaizaliton of the 
        multi-dimensional posterior
        """
        if self.n_params == 3:
            self.samples[:,2] = self.samples[:,2] * 1e5  
            fig = corner.corner(self.samples,bins=30,quantiles=[0.16, 0.5, 0.84],labels=[r"$\log N\,[\rm cm^{-3}]$",r"$b\,[\rm km/s]$", r"$z \times 10^5$"],show_titles=True,title_kwargs={"fontsize": 15})
        else:  
            fig = corner.corner(self.samples)

        pl.savefig(self.config_param.processed_product_path + '/corner_' + self.config_param.chain_short_fname + '.pdf',bbox_inches='tight',dpi=100)
        pl.clf()

    def write_model_spectrum(self):
        """
        Write to file the input segment of spectrum for fitting and the 
        best fit model flux 
        """
        output_fname = (self.config_param.processed_product_path +  
        '/spec_' + self.config_param.chain_short_fname+'.dat') 
        
        np.savetxt(output_fname,np.c_[self.config_param.wave,
                                      self.config_param.flux, 
                                      self.config_param.dflux,
                                      self.model_flux],header='wave\tflux\terror\tmodel\n')
        print('Written %s/processed_products/modelspec.dat' % 
              self.config_param.spec_path)
        
    def write_model_summary(self):
        """
        Write to file the confidence levels of all the parameters
        in the best fit model
        """
        mcmc_chain_fname = self.config_param.chain_fname + '.npy'
        output_summary_fname = self.config_param.processed_product_path + '/params_'+self.config_param.chain_short_fname+'.dat'
        write_mcmc_stats(self.config_param,output_summary_fname)


if __name__ == '__main__':

    import sys
    
    if len(sys.argv) == 1:
        print('python full_path_to_config')
    elif len(sys.argv) == 2:
        config_fname = sys.argv[1]
        print('python full_path_to_config redshift dv_range_for_plotting')
        redshift = float(raw_input('Redshift of system = '))
        dv       = float(raw_input('velocity range of spectrum plot = '))
    elif len(sys.argv) == 4:
        config_fname = sys.argv[1]
        redshift     = float(sys.argv[2]) 
        dv           = float(sys.argv[3])
    
    elif len(sys.argv) == 5:
        config_fname = sys.argv[1]
        redshift     = float(sys.argv[2]) 
        dv           = float(sys.argv[3])
        rest_wave    = float(sys.argv[4])

    else:
        print('python full_path_to_config redshift dv_range_for_plotting')
        exit()

    config_params = DefineParams(config_fname)
    output_model = ProcessModel(config_params)    
    output_model.plot_model_comparison(redshift,dv)
    output_model.write_model_summary()
    output_model.write_model_spectrum()
    output_model.plot_gr_indicator()
    output_model.corner_plot()
