###############################################################################
#
# bvp_process_model.py  (c) Cameron Liang
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Process model outputs such as corner plot, GR plot, and associated    
# ascii data files. 
#
###############################################################################


import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from bayesvp.vp_model import continuum_model_flux
from bayesvp.utilities import write_mcmc_stats, compute_burnin_GR
from bayesvp.utilities import MyParser,extrapolate_pdf,triage
from bayesvp.config import DefineParams


class ProcessModel:

    def __init__(self,config_param):

        self.config_param = config_param
        mcmc_chain_fname = config_param.chain_fname + '.npy'
        self.burnin = compute_burnin_GR(config_param.chain_fname + '_GR.dat')
        self.mcmc_chain = np.load(mcmc_chain_fname)
        self.mcmc_chain = self.mcmc_chain[self.burnin:, :, :]

        # MCMC chains
        
        self.burned_in_samples = self.mcmc_chain.reshape((-1, self.config_param.n_params))


        # Best fit model parameters and spectrum flux
        self.best_fit_params = np.median(self.burned_in_samples,axis=0)
        self.model_flux = continuum_model_flux(self.best_fit_params,config_param)

        # Get parameters labels/legends for figures
        logN_counter = 0 
        b_counter    = 0
        z_counter    = 0
        cont_param_counter = 0

        self.plot_param_labels = []
        self.gr_param_label = []
        self.ascii_filename_label = []
        for n in range(self.config_param.n_params):
            if n < self.config_param.n_params-self.config_param.cont_nparams:
                if self.config_param.vp_params_type[n] == 'logN':
                    logN_counter += 1
                    temp_label = r'$\log N_{%s}/\rm{cm^{-2}}$' % str(logN_counter)
                    self.plot_param_labels.append(temp_label)
                    self.gr_param_label.append(r'$logN_{%s}$' % str(logN_counter))
                    self.ascii_filename_label.append('logN_%s' % str(logN_counter))
                elif self.config_param.vp_params_type[n] == 'b':
                    b_counter += 1
                    temp_label = r'$b_{%s}/\rm{km s^{-1}}$' % str(b_counter)
                    self.plot_param_labels.append(temp_label)
                    self.gr_param_label.append(r'$b_{%s}$' % str(b_counter))
                    self.ascii_filename_label.append('b_%s' % str(b_counter))
                elif self.config_param.vp_params_type[n] == 'z':
                    z_counter += 1 
                    temp_label = r'$z_{%s}/10^{-5}$' % str(z_counter)
                    self.plot_param_labels.append(temp_label)
                    self.gr_param_label.append(r'$z_{%s}$' % str(z_counter))
                    self.ascii_filename_label.append('z_%s' % str(z_counter))
            else:
                cont_param_counter += 1
                temp_label = r'$a_{%s}$' % str(cont_param_counter)
                self.plot_param_labels.append(temp_label)
                self.ascii_filename_label.append(temp_label)
                self.gr_param_label.append(temp_label)

        self.plot_param_labels   = np.array(self.plot_param_labels)
        self.gr_param_label = np.array(self.gr_param_label)
        self.ascii_filename_label = np.array(self.ascii_filename_label)


    def save_marginalized_pdf(self,n):
        """Plot and save the marginalized distribution of voigt profile parameters
        """
        
        param_type  = self.config_param.vp_params_type[n]
        param_pos   = self.config_param.vp_params_flags[n]
        param_label = self.ascii_filename_label[n] 
        x = self.params_pdfs[n][0]; log_pdf = self.params_pdfs[n][1]

        # write to file
        output_name_prefix = self.config_param.data_product_path_files+'/pdf_'+ \
                            param_label +'_'+ self.config_param.chain_short_fname 
        
        f = open(output_name_prefix + '.dat','w')
        if 'z' in param_label:
            f.write('# %sx1e5\t%s\n' % (param_type, 'log10(pdf)'))
        else:
            f.write('# %s\t%s\n' % (param_type, 'log10(pdf)'))
        for i in xrange(len(x)):
            if not np.isinf(log_pdf[i]):
                f.write('%.4f\t%.16f\n' % (x[i], log_pdf[i]))
        f.close()

        print('Written %s' % output_name_prefix + '.dat')


    def spline_binned_pdf(self,bins = 30, interp_kind = 'linear'):
        """
        Convert from MCMC samples to PDF by interpolating 
        the histogram; extrapolate logN if needed. 
        """

        self.params_pdfs = []
        for n in range(self.config_param.n_params):
            bin_step_size = 0.01
            if 'z' in self.ascii_filename_label[n]:
                pdf, edges = np.histogram(self.burned_in_samples[:,n]*1e5,density=1,bins=bins)
            else:
                pdf, edges = np.histogram(self.burned_in_samples[:,n],density=1,bins=bins)

            f = interp1d(edges[1:],pdf,kind=interp_kind)
            x = np.arange(min(edges[1:]),max(edges[1:]),bin_step_size)

            if 'logN' in self.ascii_filename_label[n]:
                new_bounds = [0.0,23.0]
                x_tmp,log_pdf_tmp = extrapolate_pdf(x,f(x),new_bounds[0],new_bounds[1],
                                    bin_step_size)
                self.params_pdfs.append([x_tmp,log_pdf_tmp])

            elif 'b' in self.ascii_filename_label[n]:
                new_bounds = [0.0,500.0]
                x_tmp,log_pdf_tmp = extrapolate_pdf(x,f(x),new_bounds[0],new_bounds[1],
                                    bin_step_size)
                self.params_pdfs.append([x_tmp,log_pdf_tmp])
            else:
                min_x = np.min(x) - 100
                max_x = np.max(x) + 100
                x_tmp,log_pdf_tmp = extrapolate_pdf(x,f(x),min_x,max_x,
                                    bin_step_size)
                self.params_pdfs.append([x_tmp,log_pdf_tmp])

            if n < self.config_param.n_params-self.config_param.cont_nparams:
                self.save_marginalized_pdf(n)

        self.params_pdfs = np.array(self.params_pdfs)

    def plot_model_comparison(self,redshift,dv,central_wave=None):
        """ Plot best fit model onto spectrum for visual inspection 
        """
        c = 299792.485 # [km/s]

        if central_wave == None:
            # Use the first transition as the central wavelength
            central_wave = self.config_param.transitions_params_array[0][0][0][1]
        else:
            central_wave = float(central_wave)

        obs_spec_wave = self.config_param.wave / (1+redshift) 
        obs_spec_dv = c*(obs_spec_wave - central_wave) / central_wave
        plt.rc('text', usetex=True)

        plt.figure(1,figsize=(6,6))
        plt.step(obs_spec_dv,self.config_param.flux,'k',label=r'$\rm Data$')
        plt.step(obs_spec_dv,self.model_flux,'b',lw=2,label=r'$\rm Best\,Fit$')
        plt.step(obs_spec_dv,self.config_param.dflux,'r')
        plt.axhline(1,ls='--',c='g',lw=1.2)
        plt.axhline(0,ls='--',c='g',lw=1.2)
        plt.axvline(0,ls='--',c='g',lw=1.2)
        plt.ylim([-0.1,1.4])
        plt.xlim([-dv,dv])
        plt.xlabel(r'$dv\,[\rm km/s]$',fontsize=15)
        plt.ylabel(r'$\rm Normalized\,Flux$',fontsize=15)
        plt.legend(loc='best')
        
        output_name = (self.config_param.data_product_path_plots + '/modelspec_'
                      + self.config_param.chain_short_fname + '.pdf')
        plt.savefig(output_name,bbox_inches='tight',dpi=100)
        plt.clf()
        print('Written %s' % output_name)

    def corner_plot(self,nbins=30,fontsize=11,cfigsize=[6,6],truths=None):
        """
        Make triangle plot for visuaizaliton of the 
        multi-dimensional posterior
        """

        self.truths = truths
        if self.truths:
            self.truths = np.array(truths)
            if len(truths) != self.config_param.n_params:
                sys.exit('Number of true values (%i) should equal number of parameters (%i)\n Exiting program...' 
                        % (len(truths),self.config_param.n_params))

        for n in range(self.config_param.n_params):
            if (n < (self.config_param.n_params-self.config_param.cont_nparams) 
                and self.config_param.vp_params_type[n] == 'z'):
                self.burned_in_samples[:,n] = self.burned_in_samples[:,n] * 1e5
                
                if self.truths:
                    self.truths[n] = self.truths[n] * 1e5

        plt.figure(1)
        output_name = self.config_param.data_product_path_plots + '/corner_' + \
                      self.config_param.chain_short_fname + '.pdf'
        weights_of_chains = np.ones_like(self.burned_in_samples)

        fig = triage(self.burned_in_samples,weights_of_chains,
                    self.plot_param_labels,figsize=cfigsize,nbins=nbins,
                    figname=output_name,fontsize=fontsize,labelsize=10)

        plt.clf()

        print('Written %s' % output_name)

    def plot_gr_indicator(self):
        """
        Make plot for the evolution of GR indicator 
        as a function of steps
        """
        gr_fname = self.config_param.chain_fname + '_GR.dat'
        data = np.loadtxt(gr_fname,unpack=True)
        steps = data[0]; grs = data[1:]
        
        plt.figure(1,figsize=(6,6))
        for i in xrange(len(grs)):
            plt.plot(steps,grs[i],lw=1.5,label=self.gr_param_label[i])
            
        plt.legend(loc='best')
        plt.xscale('log')

        plt.xlabel(r'$N(\rm{steps})$',fontsize=15)
        plt.ylabel(r'$R_{\rm GR}$',fontsize=15)

        output_name = self.config_param.data_product_path_plots + '/GR_' + \
                      self.config_param.chain_short_fname + '.png'
        plt.savefig(output_name,bbox_inches='tight',dpi=100)
        plt.clf()

        print('Written %s' % output_name)


    def write_model_spectrum(self):
        """
        Write to file the input segment of spectrum for fitting and the 
        best fit model flux 
        """
        output_fname = self.config_param.data_product_path_files +  \
                    '/spec_' + self.config_param.chain_short_fname+'.dat' 
        
        np.savetxt(output_fname,np.c_[self.config_param.wave,
                                      self.config_param.flux, 
                                      self.config_param.dflux,
                                      self.model_flux],
                                      header='wave\tflux\terror\tmodel\n')
        print('Written %s' % output_fname)
        
    def write_model_summary(self):
        """
        Write to file the confidence levels of all the parameters
        in the best fit model
        """
        mcmc_chain_fname = self.config_param.chain_fname + '.npy'
        output_summary_fname = self.config_param.data_product_path_files + \
                        '/params_'+self.config_param.chain_short_fname+'.dat'
        write_mcmc_stats(self.config_param,output_summary_fname)


def main():

    parser = MyParser()
    parser.add_argument('config_fname',help="full path to config filename", nargs='?',type=str)
    parser.add_argument('redshift',help="central redshift for plotting spectrum", nargs='?',type=float)
    parser.add_argument('dv',help="velocity range for spectrum (+/-dv)", nargs='?',type=float)
    #parser.add_argument("--truths", nargs='+', type=float)
    parser.add_argument('-t', "--test",help="plot data product of test", action="store_true")    
    
    args = parser.parse_args()

    if len(sys.argv)<4 and not args.test:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.test:
        from bayesvp.utilities import get_bayesvp_Dir
        path = get_bayesvp_Dir()
        config_fname = path + '/data/example/config_OVI.dat'

        config_param = DefineParams(config_fname)
        args.redshift = 0.0; args.dv = 300.0
    
    else:
        if os.path.isfile(args.config_fname):
            config_param = DefineParams(args.config_fname)
        else:
            sys.exit('Config file does not exist:\n %s' % args.config_fname)

    output_model = ProcessModel(config_param)
    output_model.spline_binned_pdf()
    output_model.plot_model_comparison(args.redshift,args.dv)
    output_model.write_model_summary()
    output_model.write_model_spectrum()
    output_model.plot_gr_indicator()
    #output_model.corner_plot(nbins=30,truths=args.truths)
    output_model.corner_plot(nbins=30)


if __name__ == '__main__':
    sys.exit(main() or 0)

