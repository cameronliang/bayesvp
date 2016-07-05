import numpy as np
import pylab as pl
import sys,os,re 

class obs_data:
    """
    Read and define fitting parameters from 
    config file  
    """
    
    def __init__(self,config_fname):
        self.config_fname = config_fname
        
        # Read and filter empty lines
        all_lines = filter(None,(line.rstrip() for line in open(config_fname)))

        # Remove commented lines
        self.lines = []
        for line in all_lines:
            if not line.startswith('#') and not line.startswith('!'): 
                self.lines.append(line)    

    def fileio_mcmc_params(self):
        # Paths and fname strings
        self.spec_path         = self.lines[0].rstrip()
        self.chain_short_fname = self.lines[1].rstrip()

        self.mcmc_outputpath   = self.spec_path + '/vpfit_mcmc'
        if not os.path.isdir(self.mcmc_outputpath):
		    os.mkdir(self.mcmc_outputpath)
        self.chain_fname = self.mcmc_outputpath + '/' + self.chain_short_fname

        # MCMC related
        self.nwalkers     = int(self.lines[2].rstrip().split(' ')[0])
        self.nsteps       = int(self.lines[2].rstrip().split(' ')[1])
        self.threads      = int(self.lines[2].rstrip().split(' ')[2])
 

    def fitting_data(self): 
         
        for line in self.lines: 
            if re.search('%%',line):
                spec_fname_line = line

        spec_data_array = spec_fname_line.split(' ')
        self.spec_fname = self.spec_path + '/' + spec_data_array[1]

        # Select spectral range to fit
        wavelength_sections = []
        if len(spec_data_array[2:]) % 2 != 0:
            print('There is odd number of wavelengths entered in config file')
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

        all_inds = []
        for i in range(len(self.wave_begins)):
            inds = np.where((wave>=self.wave_begins[i]) & (wave<self.wave_ends[i]))[0]
            all_inds.append(inds)
        
        all_inds   = np.hstack(np.array(all_inds))
        self.wave  = wave[all_inds]
        self.flux  = flux[all_inds]
        self.dflux = dflux[all_inds]

    def fitting_params(self):
        amu = 1.66053892e-24   # 1 atomic mass in grams
        data_file = './data/atom.dat'
        atoms  = np.loadtxt(data_file, dtype=str, usecols=[0])
        states = np.loadtxt(data_file, dtype=str, usecols=[1])
        wave   = np.loadtxt(data_file, usecols=[2])
        osc_f  = np.loadtxt(data_file, usecols=[3])
        gamma  = np.loadtxt(data_file, usecols=[4])
        mass   = np.loadtxt(data_file, usecols=[5]) * amu 

        def get_transitions_params(atom,state,wave_start,wave_end,redshift):

            inds = np.where((atoms == atom) & 
                            (states == state) & 
                            (wave >= wave_start/(1+redshift)) & 
                            (wave < wave_end/(1+redshift)))[0]
            if len(inds) == 0:
                return None
            else:
                transitions_params_array = np.array([osc_f[inds],wave[inds],gamma[inds], mass[inds]])
                return np.transpose(transitions_params_array)
        
        # Lines in config file that contain the component parameters
        # i.e atom, state, logN, b, z
        component_lines = []
        for line in self.lines: 
            if re.search('%',line):
                component_lines.append(line)
        component_lines = component_lines[1:] 

        logNs = []; bs = []; redshifts = []
        transitions_params_array = []
        guess_alpha = []
        for line in component_lines:
            line = filter(None,line.split(' '))
            atom  = line[1]; state = line[2] # To obtain transition data
            logNs.append(float(line[3])); 
            bs.append(float(line[4])); 
            redshifts.append(float(line[5])) 
            
            for i in xrange(len(self.wave_begins)):
                # Each component gets a set of all of the transitions data
                temp_transitions_params = get_transitions_params(atom,state,self.wave_begins[i],self.wave_ends[i],float(line[5]))
                if temp_transitions_params is not None:
                    transitions_params_array.append(temp_transitions_params)
        
        
        # Shape = (n_components,n_transitions,4)
        # For each component, there is n_transitions within the wavelength 
        # regime, and each transition has (wave0,f_jk,gamma,mass) properties.
 
        # A component is defined vaguely as number of sets of (logN, b, z)
        # They also need to fall within range of specified wavelength (guessed z cannot be too large)
        self.transitions_params_array = np.array(transitions_params_array)
        self.vp_params = np.array([logNs,bs,redshifts]).T
        self.n_component = len(transitions_params_array)        
         

"""Global Object defined by the class""" 

# Command line argument - run time feed. e.g., python main.py full_path_to_config_file
config_fname = sys.argv[1]
obs_spec = obs_data(config_fname)
obs_spec.fileio_mcmc_params()
obs_spec.fitting_data()
obs_spec.fitting_params()

print('\n')
print('Spec Path: %s'     % obs_spec.spec_path)
print('Fitting:')
for i in xrange(len(obs_spec.transitions_params_array)):
    for j in xrange(len(obs_spec.transitions_params_array[i])):
        print('   Transitions Wavelength: %f' % obs_spec.transitions_params_array[i][j][1])
for i in xrange(len(obs_spec.wave_begins)):
    print('Selected data wavelegnth region: (%.3f, %.3f)' % (obs_spec.wave_begins[i],obs_spec.wave_ends[i]))
print('Output MCMC chain: %s' % obs_spec.chain_short_fname) 
print('\n')
