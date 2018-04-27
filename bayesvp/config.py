################################################################################
#
# config.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Read in and define all of the parameters for voigt profile based on a
# config file; see ./scripts/bvp_write_config.py for producing config file.
# or example in ./data/example/
################################################################################

import numpy as np
import os
import re
import sys
import ntpath

from bayesvp.utilities import get_transitions_params, MyParser


class DefineParams:
    """
    Read and define fitting parameters from 
    config file

    Attributes:
    -----------
    lines: array_like
        All lines containing in config file
    spec_path: str
        Full path to the spectrum file 
    chain_short_fname: str
        Name of the output mcmc chain (with .npy extension)
    self_bvp_test: bool
        True if it is a test run
    chain_fname: str
    output_path: str
    mcmc_outputpath: str
        output path for the MCMC chain
    data_product_path: str
        parent directory for data output
    data_product_path_files: str
        output path for ascii files such as best fits and confidence 
        levels
    data_product_path_plots: str
        output path for corner plots, model comparison and etc
    nwalkers: int
        Number of walkers
    nsteps: int
        Number of steps for each walker
    nthreads: int
        Number of parallel threads
    wave: array
        Selected region of the input spectral data
    flux: array
        flux of the spectrum
    error: array
        uncertainty of flux of the input spectrum
    wave_begins: array_like
        Selected wavelength regions bounds
    wave_ends: array_like
        Selected wavelength regions bounds
    vp_params: array_like
    transitions_params_array: array_like
    vp_params_type: array_like
        Voigt profile model parameter types, i.e [logN, b, z]
    vp_params_flags: array_like
        indexed flags that indicate the parameter type
    priors: array_like
        Priors for three types of parameters (logN, b, z)
        shape = (3,2)
    n_component: int
        Number of Vogit components defined for the model
    lsf: array_like
        Line spread function to be convolved with the model
    cont_normalize: bool
        True if user choose to include continuum fit
    cont_nparams: int
        Number of parameters from the polynomial continuum model
    cont_prior: array_like
        All continuum parameters are limited by +/- this value
    """

    def __init__(self,config_fname):

        self.config_fname = config_fname
        self.config_basename = ntpath.basename(config_fname)
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
        
        # continuum model preset to false
        self.cont_normalize  = False
        self.cont_nparams = 0
        self.cont_prior   = 1.0
        self.self_bvp_test = False

        # Paths and fname strings
        for line in self.lines:
            line = filter(None,line.split(' '))

            if 'spec_path' in line or 'input' in line or 'spectrum' in line:
                if line[1] == 'test_path_to_spec':
                    self.spec_path = (os.path.dirname(os.path.abspath(__file__)) + 
                                     '/data/example')
                    self.self_bvp_test = True

                else:
                    self.spec_path = line[1]

            elif 'output' in line or 'chain' in line:
                self.chain_short_fname = line[1]

            elif 'continuum' in line or 'contdegree' in line:
                self.cont_normalize  = True
                self.cont_nparams = int(line[1]) + 1 # n_param = poly degree + 1 (offset)

            elif 'cont_prior' in line or 'contprior' in line:
                if self.cont_normalize and self.cont_nparams>0:
                    tmp_priors = [float(i) for i in line[1:]]
                    # reverse direction to match order of cont params
                    # {a_i} in polynomial a0*x^0 + a1*x^1 + ....
                    tmp_priors = tmp_priors[::-1]
                    if len(tmp_priors) == 1:
                        # all parameters share the same prior
                        self.cont_prior = np.ones(self.cont_nparams)*tmp_priors
                    elif len(tmp_priors) == self.cont_nparams:
                        # each have its unique prior
                        self.cont_prior = np.array(tmp_priors)
                    else:
                        sys.exit('Please enter only 1 continuum prior or match'
                                ' the number of continuum parameters. Exiting program..')
                else:
                    sys.exit('Continuum fit is not set or degree is less than 0.\n')

            elif 'mcmc_params' in line or 'mcmc' in line:
                self.nwalkers = int(line[1])
                self.nsteps   = int(line[2])
                self.nthreads = int(line[3])

                # Default
                self.model_selection = 'bic'      
                self.mcmc_sampler    = 'kombine'

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
            sys.exit('There is an odd number of wavelengths entered in config file.\n Exiting program...')
        else:
            self.wave_begins = np.array(spec_data_array[2:][0::2]).astype(float)
            self.wave_ends   = np.array(spec_data_array[2:][1::2]).astype(float)

            for i in range(len(self.wave_begins)):
                if self.wave_begins[i] >= self.wave_ends[i]:
                    sys.exit('Starting wavelength cannot be greater or equal to ending wavelength: (%.3f, %.3f); exiting program...' 
                            % (self.wave_begins[i] ,self.wave_ends[i]))

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
        inds = np.where((self.flux < 0)); self.flux[inds] = 0
        inds = np.where((self.dflux < 0)); self.dflux[inds] = 0

        if len(self.wave) == 0 or len(self.flux) == 0 or len(self.dflux) == 0:
            raise SystemExit('No data within specified wavelength range.' \
                             'Please check config file and spectrum.')
            

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
        for i in range(len(component_lines)):
            line = component_lines[i]
            line = filter(None,line.split(' '))

            atom  = line[1]; state = line[2] # To obtain transition data
            logNs.append(line[3])
            bs.append(line[4])
            redshifts.append(line[5])

            if line[5][-1].isalpha():
                self.redshift = line[5][:-1]
            else:
                self.redshift = line[5]
                

            transitions_params_array.append([])
            # Each component gets a set of all of the transitions data
            for j in range(len(self.wave_begins)):
                # each wavelength regions gets all of the transitions
                temp_params = get_transitions_params(atom,state,self.wave_begins[j],self.wave_ends[j],float(self.redshift))
                transitions_params_array[i].append(temp_params)
        
        # Shape = (n_component,n_regions,n_transitions,4) 
        self.transitions_params_array = np.array(transitions_params_array)
        self.vp_params = np.transpose(np.array([logNs,bs,redshifts]))
        self.n_component = len(component_lines) 

        # Define what kind of parameters to get walker initiazation ranges.
        # and for fixing and freeing paramters. 
        
        # Note that this assumed the pattern of parameters. 
        # will update for continuum parameters. 
        vp_params_type = [None]*len(self.vp_params.flatten())
        vp_params_type[::3]  = ['logN'] * (len(vp_params_type[::3]))
        vp_params_type[1::3] = ['b']    * (len(vp_params_type[1::3]))
        vp_params_type[2::3] = ['z']    * (len(vp_params_type[2::3]))
        

        flat_params = self.vp_params.flatten()
        flags = np.zeros(len(flat_params))

        letters = [None]*len(flat_params)
        for i in range(len(flat_params)):
            for j in range(len(flat_params[i])):
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
            self.n_params = self.n_params + self.cont_nparams

        # Make directories for data products
        if self.self_bvp_test:
            # write to local direcotry if it is test to avoid permission issues in bayesvp library location
            self.output_path = '.' + '/bvp_output_z' + str(self.redshift)

        else:
            self.output_path = self.spec_path + '/bvp_output_z' + str(self.redshift)

        self.mcmc_outputpath = self.output_path + '/chains'
        
        self.data_product_path = self.output_path +'/data_products'
        self.data_product_path_files = self.data_product_path + '/ascii'
        self.data_product_path_plots = self.data_product_path + '/plots'
        self.chain_fname = self.mcmc_outputpath + '/' + self.chain_short_fname

        try: 
            os.makedirs(self.mcmc_outputpath)
        except OSError:
            if not os.path.isdir(self.mcmc_outputpath):
                raise

        try: 
            os.makedirs(self.data_product_path_files)
        except OSError:
            if not os.path.isdir(self.data_product_path_files):
                raise

        try: 
            os.makedirs(self.data_product_path_plots)
        except OSError:
            if not os.path.isdir(self.data_product_path_plots):
                raise

        ########################################################################
        # Determine the LSF by specifying LSF filename with 
        # 'database' directory under self.spec_path.    
        # Assumes LSF file contains only 1 column of data
        # -----------
        # lsf 
        ########################################################################

        # Check if LSF is specified in config file
        defined_lsf = False
        for line in self.lines:
            if re.search('lsf',line) or re.search('LSF',line):
                lsf_line = line.split(' ')[1:]
                defined_lsf = True
                if not os.path.isdir(self.spec_path + '/database'):
                    os.mkdir(self.spec_path + '/database')
                    sys.exit('Require LSF file to be in %s' % self.spec_path + '/database\n Exiting program...')
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
                sys.exit('Please check if number of LSF matches wavelength regions. Exiting program...')
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
                if len(line) != 3:
                    sys.exit('Error! In config file, format for logN prior:\n logN min_logN max_logN\nExiting program...')
                self.priors[0] = [float(line[1]),float(line[2])]

            if 'b' in line:
                if len(line) != 3:
                    sys.exit('Error! In config file, format for b prior:\n z min_b max_b\nExiting program...')
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
                    sys.exit('Error! In config file, format for z prior:\n z center_z |min_dv| |max_dv|\nor\n' + 
                            ' z center_z dv\nThe latter option will use +/- dv [km/s]')
                self.priors[2] = [min_z,max_z]
    
    def print_config_params(self):

        # First copy the original config file to output path
        # but replace the `test_path_to_spec` to actual output path
        with open(self.output_path + '/' + self.config_basename,'w') as f_config:
            # Paths and fname strings
            for line in self.lines:
                tmp_line = filter(None,line.split(' '))

                if 'spec_path' in tmp_line or 'input' in tmp_line or 'spectrum' in tmp_line:
                    if tmp_line[1] == 'test_path_to_spec':
                        cwd = os.getcwd()
                        print cwd + self.output_path[1:]
                        f_config.write('spec_path %s\n' % (cwd + self.output_path[1:]))
                else:
                    f_config.write('%s\n' % line)
        
        # Also copy the spectrum to the output path in the case of a test
        import shutil
        shutil.copy(self.spec_fname,self.output_path)

        f_logging = open(self.output_path + '/config.log','w')

        f_logging.write('Config file: %s\n'     % self.config_fname)
        f_logging.write('Spectrum Path: %s\n'     % self.spec_path)
        f_logging.write('Spectrum name: %s\n'     % self.spec_short_fname)
        
        f_logging.write('Fitting %i components with transitions:\n' % self.n_component)
        for i in range(len(self.transitions_params_array)):
            for j in range(len(self.transitions_params_array[i])):
                if not np.isnan(self.transitions_params_array[i][j]).any():
                    for k in range(len(self.transitions_params_array[i][j])):
                        rest_wavelength = self.transitions_params_array[i][j][k][1]
                        f_logging.write('    Transitions Wavelength: %.3f\n' % rest_wavelength)
                else:
                    sys.exit('No transitions satisfy the wavelength regime for fitting;Check input wavelength boundaries')

        f_logging.write('Selected data wavelegnth region:\n')
        for i in range(len(self.wave_begins)):
            f_logging.write('    [%.3f, %.3f]\n' % (self.wave_begins[i],self.wave_ends[i])) 
        
        f_logging.write('MCMC Sampler: %s\n' % self.mcmc_sampler)
        f_logging.write('Model selection method (if needed): %s\n' % self.model_selection)
        f_logging.write('Walkers,steps,threads : %i,%i,%i\n' % (self.nwalkers,self.nsteps,self.nthreads))
        f_logging.write('Priors: ')
        f_logging.write('logN: [min,   max] = [%.3f, %.3f]\n' % (self.priors[0][0],self.priors[0][1]))
        f_logging.write('b:            [min,   max] = [%.3f, %.3f]\n' % (self.priors[1][0],self.priors[1][1]))
        f_logging.write('redshift:     [min,   max] = [%.5f, %.5f]\n' % (self.priors[2][0],self.priors[2][1]))
        
        if self.cont_normalize:
            f_logging.write('Continuum polynomial degree: %i\n'     % (self.cont_nparams-1))
            f_logging.write('Continuum priors with +/- a_i: ')
            for i in range(len(self.cont_prior)):
                f_logging.write('%f\t' % self.cont_prior[i])
            f_logging.write('\n')
        
        f_logging.close()
    
def main():
    parser = MyParser()
    parser.add_argument('config_fname',help="full path to config filename", nargs='?')
    parser.add_argument("-t", "--test",help="test for reading config file",
                        action="store_true")
    parser.add_argument("-v", "--verbose",help="print config summary to file ",
                        action="store_true")
    

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()

    if args.test:
        from bayesvp.utilities import get_bayesvp_Dir
        path = get_bayesvp_Dir()
        config_fname = path + '/data/example/config_OVI.dat'
        config_params = DefineParams(config_fname)
        config_params.print_config_params()

    if args.config_fname:
        if os.path.isfile(args.config_fname):
            config_params = DefineParams(args.config_fname)
            if args.verbose:
                config_params.print_config_params()
        else:
            sys.exit('Config file does not exist:\n %s' % args.config_fname)

if __name__ == '__main__':
    sys.exit(main() or 0)
