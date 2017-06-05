################################################################################
#
# Config.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Read in and define all of the parameters for voigt profile based on a  
# config file; see write_config.py for producing config file.     
################################################################################

import numpy as np
import os
import re 

from Utilities import get_transitions_params,printline


class DefineParams:
    """
    Read and define fitting parameters from 
    config file

    Attributes:
    -----------
    lines, spec_path, chain_short_fname, 
    chain_fname, mcmc_outputpath: str
        Files I/O
    nwalkers, nsteps, nthreads: int    
        MCMC parameters
    wave,flux,error: array
        Selected region of the input spectral data
    wave_begins, wave_ends: array_like
        Selected wavelength regions bounds
    vp_params, transitions_params_array, vp_params_type,vp_params_type:array_like
        Model auxilary parameters/controls 
    n_component: int
        Number of Components defined by the model
    lsf: array_like
        Line spread function to be convolved with the model
    priors: array_like; shape = (3,2)
        Priors for three types of parameters (logN, b, z)

    """

    def __init__(self,config_fname):
        self.config_fname = config_fname
        printline()
        print('Config file: %s' % config_fname)
        
        # Read and filter empty lines
        all_lines = filter(None,(line.rstrip() for line in open(config_fname)))

        # Remove commented lines
        self.lines = []
        for line in all_lines:
            if not line.startswith('#') and not line.startswith('!'): 
                self.lines.append(line)    

        
        ########################################################################
        # Retrieve MCMC parameters from config file
        # --------
        # self.spec_path, self.chain_fname, self.nwalkers, self.nsteps,
        # self.nthreads
        ########################################################################
        # Paths and fname strings
        for line in self.lines:
            line = filter(None,line.split(' '))
            if 'spec_path' in line or 'input' in line or 'spectrum' in line:
                if line[1] == 'test_path_to_spec':
                    self.spec_path = os.path.dirname(os.path.abspath(__file__)) + '/tests/'
                else:
                    self.spec_path = line[1]
            elif 'output' in line or 'chain' in line:
                self.chain_short_fname = line[1]
            elif 'mcmc_params' in line or 'mcmc' in line:
                self.nwalkers = int(line[1])
                self.nsteps   = int(line[2])
                self.nthreads = int(line[3])

                # Default
                self.model_selection = 'bic'      
                self.mcmc_sampler    = 'kombine'
                self.cont_normalize  = False

                # Set up true by a flag in config 
                self.cont_normalize  = True
                # Change keys if defined in config 
                for key in line[3:]:
                    if key in ['kombine','emcee']:
                        self.mcmc_sampler = key
                    elif key in ['aic','bic','bf']:
                        self.model_selection = key  
        
        ########################################################################
        # Get the spectral data specified by the config file
        # The spectrum file assumes three column of data with 
        # [wave,flux,error]
        # ---------
        # self.spec_short_fname, self.spec_fname 
        # self.wave, self.flux, self.error 
        ########################################################################
        for line in self.lines: 
            if re.search('%%',line):
                spec_fname_line = line

        spec_data_array = spec_fname_line.split(' ')
        self.spec_short_fname = spec_data_array[1]
        self.spec_fname = self.spec_path + '/' + spec_data_array[1]

        # Select spectral range to fit
        if len(spec_data_array[2:]) % 2 != 0:
            print('There is an odd number of wavelengths entered in config file.')
            print('Exiting program...')
            exit()
        else:
            self.wave_begins = np.array(spec_data_array[2:][0::2]).astype(float)
            self.wave_ends   = np.array(spec_data_array[2:][1::2]).astype(float)

            for i in xrange(len(self.wave_begins)):
                if self.wave_begins[i] >= self.wave_ends[i]:
                    print('Starting wavelength cannot be greater or equal to ending wavelength: (%.3f, %.3f); exiting program...' % (self.wave_begins[i] ,self.wave_ends[i]))
                    exit()

        wave,flux,dflux = np.loadtxt(self.spec_fname,
                                    unpack=True,usecols=[0,1,2])
        
        # Select regions of interests 
        all_inds = []
        for i in range(len(self.wave_begins)):
            inds = np.where((wave>=self.wave_begins[i]) & (wave<self.wave_ends[i]))[0]
            all_inds.append(inds)

        all_inds   = np.hstack(np.array(all_inds))
        wave = wave[all_inds]; flux = flux[all_inds]; dflux = dflux[all_inds]
        
        # Remove NaN pixels in flux
        inds = np.where((~np.isnan(flux)))
        self.wave = wave[inds]; self.flux = flux[inds]; self.dflux = dflux[inds]

        # Set negative pixels in flux and error 
        inds = np.where((self.flux < 0)); self.flux[inds] = 0; 
        inds = np.where((self.dflux < 0)); self.dflux[inds] = 0;

        if len(self.wave) == 0 or len(self.flux) == 0 or len(self.dflux) == 0:
            print('No data within specified wavelength range. Please check config file and spectrum.')
            exit()

        ########################################################################
        # Get Voigt profile parameters of arbitary number of components
        # specified in the config file.

        # Uses './data/atom.dat' to read in atomic/transition data with format:
        # [atom state rest_wavelength oscillator_strength damping_coeff mass_amu]
        # Users can add additional row to the file for new atomic data

        # --------
        # self.vp_params, self.transitions_params_array, self.vp_params_flags, 
        # self.vp_params_type, self.n_component 
        ########################################################################
        

        # Lines in config file that contain the component parameters
        # i.e atom, state, logN, b, z
        component_lines = []
        for line in self.lines: 
            if re.search('%',line):
                component_lines.append(line)
        component_lines = component_lines[1:] 


        logNs = []; bs = []; redshifts = []
        transitions_params_array = []
        for i in xrange(len(component_lines)):
            line = component_lines[i]
            line = filter(None,line.split(' '))

            atom  = line[1]; state = line[2] # To obtain transition data
            logNs.append(line[3]); 
            bs.append(line[4]);
            redshifts.append(line[5])

            if line[5][-1].isalpha():
                self.redshift = line[5][:-1]
            else:
                self.redshift = line[5]
                

            transitions_params_array.append([])
            # Each component gets a set of all of the transitions data
            for j in xrange(len(self.wave_begins)):
                # each wavelength regions gets all of the transitions
                temp_params = get_transitions_params(atom,state,self.wave_begins[j],self.wave_ends[j],float(self.redshift))
                transitions_params_array[i].append(temp_params)
        
        # Shape = (n_component,n_regions,n_transitions,4) 
        self.transitions_params_array = np.array(transitions_params_array)
        self.vp_params = np.array([logNs,bs,redshifts]).T
        self.n_component = len(component_lines) 


        # Define what kind of parameters to get walker initiazation ranges.
        # and for fixing and freeing paramters. 
        vp_params_type = [None]*len(self.vp_params.flatten())
        vp_params_type[::3]  = ['logN'] * (len(vp_params_type[::3]))
        vp_params_type[1::3] = ['b']    * (len(vp_params_type[1::3]))
        vp_params_type[2::3] = ['z']    * (len(vp_params_type[2::3]))
        

        flat_params = self.vp_params.flatten()
        flags = np.zeros(len(flat_params))
        free_params = np.zeros(len(flat_params))

        letters = [None]*len(flat_params)
        for i in xrange(len(flat_params)):
            for j in xrange(len(flat_params[i])):
                if flat_params[i][j].isalpha():
                    letters[i] = flat_params[i][j]
        unique_letters = filter(None,list(set(letters)))

        n_free_params_counter = 0
        for i in range(len(letters)):
            if letters[i] == None:
                flags[i] = n_free_params_counter
                n_free_params_counter += 1

        for unique_letter in unique_letters:
            inds = [i for i, x in enumerate(letters) if x == unique_letter]
            if unique_letter.islower(): 
                flags[inds] = n_free_params_counter
                n_free_params_counter += 1
            else:
                for index in inds:
                    flags[index] = None

        # Model uses these to construct sets of (logN, b, z) for each component
        self.vp_params_type  = np.array(vp_params_type)
        self.vp_params_flags = np.array(flags)
        self.n_params        = n_free_params_counter
        if self.cont_normalize:
            self.n_params = self.n_params + 2
        

        # Make directories for data products
        self.mcmc_outputpath = self.spec_path+'/bvp_chains_' + str(self.redshift)
        if not os.path.isdir(self.mcmc_outputpath):
		    os.mkdir(self.mcmc_outputpath)
        self.chain_fname = self.mcmc_outputpath + '/' + self.chain_short_fname

        self.processed_product_path = self.spec_path+'/processed_products_' + str(self.redshift)
        if not os.path.isdir(self.processed_product_path):
		    os.mkdir(self.processed_product_path)



        """
        ########################################################################
        Determine the LSF by specifying LSF filename with 
        'database' directory under self.spec_path.    
        Assumes LSF file contains only 1 column of data
        -----------
        lsf 
        ########################################################################
        """
        # Check if LSF is specified in config file
        defined_lsf = False
        for line in self.lines:
            if re.search('lsf',line) or re.search('LSF',line):
                lsf_line = line.split(' ')[1:]
                defined_lsf = True
                if not os.path.isdir(self.spec_path + '/database'):
                    os.mkdir(self.spec_path + '/database')
                    print('Require LSF file to be in %s' % self.spec_path + '/database')
                    exit()
                break

        # Get the LSF function from directory 'database'
        if defined_lsf:
            if len(lsf_line) == len(self.wave_begins):
                self.lsf = []
                for lsf_fname in lsf_line:
                    # assume lsf file has one column 
                    fname = self.spec_path + '/database/' + lsf_fname
                    self.lsf.append(np.loadtxt(fname))
            elif len(lsf_line) == 1:
                for lsf_fname in lsf_line:
                    # assume lsf file has one column 
                    fname = self.spec_path + '/database/' + lsf_fname
                    self.lsf = np.loadtxt(fname)
            else:
                print('Please check if number of LSF mataches wavelength regions; exiting..')
                exit()
        else:
            # Convolve with LSF = 1
            self.lsf = np.ones(len(self.wave_begins))

        
        #######################################################################
        # Read priors and use them for walker initialization 
        # 
        
        # format in config file:
        # logN min_logN max_logN
        # b    min_b    max_b
        # z    center_z min_dv max_dv (range defined by range of velocity) [km/s]  
        # -----------
        # self.priors
        #######################################################################
        
        self.priors = np.zeros((3,2))
        for line in self.lines:
            line = np.array(line.split(' '))
            line = filter(None,line)
            if 'logN' in line:
                self.priors[0] = [float(line[1]),float(line[2])]
            if 'b' in line:
                self.priors[1] = [float(line[1]),float(line[2])]
            if 'z' in line:
                c = 299792.458 # speed of light [km/s]
                if len(line) == 4:
                    center_z,min_dv,max_dv = float(line[1]),float(line[2]),float(line[3])
                    if min_dv == max_dv:
                        min_dv = -min_dv
                    min_z,max_z = center_z+min_dv/c,center_z+max_dv/c
                elif len(line) == 3:
                    center_z,dv = float(line[1]),float(line[2])
                    min_z,max_z = center_z-dv/c,center_z+dv/c
                else:
                    print('In config, format for z:')
                    print('z center_z |min_dv| |max_dv|')
                    exit()
                self.priors[2] = [min_z,max_z]
    def print_config_params(self):
        printline()
        print('Config file: %s'     % self.config_fname)
        print('Spectrum Path: %s'     % self.spec_path)
        print('Spectrum name: %s'     % self.spec_short_fname)
        print('Fitting %i components with transitions: ' % self.n_component)
        for i in xrange(len(self.transitions_params_array)):
            for j in xrange(len(self.transitions_params_array[i])):
                if not np.isnan(self.transitions_params_array[i][j]).any():
                    for k in xrange(len(self.transitions_params_array[i][j])):
                        rest_wavelength = self.transitions_params_array[i][j][k][1]
                        print('    Transitions Wavelength: %.3f' % rest_wavelength)
                else:
                    print('No transitions satisfy the wavelength regime for fitting;Check input wavelength boundaries')
                    exit()

        print('Selected data wavelegnth region:')
        for i in xrange(len(self.wave_begins)):
            print('    [%.3f, %.3f]' % (self.wave_begins[i],self.wave_ends[i])) 
        print('MCMC Sampler: %s' % self.mcmc_sampler)
        print('Model selection method: %s' % self.model_selection)
        print('Walkers,steps,threads : %i,%i,%i' % (self.nwalkers,self.nsteps,self.nthreads))
        print('Priors: ')
        print('logN:     [min,   max] = [%.3f, %.3f] ' % (self.priors[0][0],self.priors[0][1]))
        print('b:        [min,   max] = [%.3f, %.3f] ' % (self.priors[1][0],self.priors[1][1]))
        print('redshift: [min,   max] = [%.5f, %.5f] ' % (self.priors[2][0],self.priors[2][1]))
        print('\n')

    def plot_spec(self):
        import matplotlib.pyplot as pl
        pl.step(self.wave,self.flux,color='k')
        pl.step(self.wave,self.dflux,color='r')
        

if __name__ == '__main__':

    # test
    import sys
    config_fname = sys.argv[1]
    
    # Load config parameter object 
    obs_spec = DefineParams(config_fname)    
    obs_spec.print_config_params()
