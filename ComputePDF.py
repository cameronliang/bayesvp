import numpy as np
import pylab as pl
import sys,os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from Config import DefineParams

def derivative_cdf(x,y):
    dx=x[1:] - x[:-1]
    diff = (y[1:]-y[:-1]) /dx
    return diff

def extract_chain(mcmc_chain_fname,para_name):
    """
    This assumes simple single component model
    with parameter [logN, b, z]
    """    
    burnin_fraction_chain = 0.5
    chain = np.load(mcmc_chain_fname)
    burnin  = int(np.shape(chain)[0]*burnin_fraction_chain)
    my_dict = {'logN':0, 'b':1,'z':2}
    col_num = my_dict[para_name]
    
    # return 1D chain of parameter x 
    return chain[burnin:,:,col_num].flatten()

def truncated_pdf(x,pdf,x_stepsize,left_boundary_x,right_boundary_x):
    """ 
    Smoothing of the CDF can create negative 
    derivative of CDF (e.g., PDF); Chop off the ends 
    where it becomes negative probability 
    """
     
    # remove all the NaNs. 
    good_inds = np.where(~np.isnan(pdf))[0]
    old_pdf = pdf; old_x = x; 
    pdf  = pdf[good_inds]
    x = x[good_inds]

    # Get rid of all the negative indices
    negative_pdf_inds = np.where(pdf <0)[0]
    if len(negative_pdf_inds) > 0:
        peak_inds = np.where(pdf == max(pdf))[0]
        left_bound_ind = 0; right_bound_ind = len(pdf)
        
        # Get the left side index that is negative pdf
        if len(np.where(negative_pdf_inds<peak_inds)[0])>0:
            left_bound_ind  = max(negative_pdf_inds[negative_pdf_inds<peak_inds])
        # Get the right side index that is negative pdf
        if len(np.where(negative_pdf_inds>peak_inds)[0])>0:
            right_bound_ind = min(negative_pdf_inds[negative_pdf_inds>peak_inds])
               
        # select all the non-negative elements
        pdf  = pdf[left_bound_ind+1:right_bound_ind]
        x = x[left_bound_ind+1:right_bound_ind]
        

    log_pdf = np.log10(pdf)
    min_x = min(x); max_x = max(x);
    
    
    entered_left_condition = False
    if min_x > left_boundary_x:

        # equation of a line with +10 slope going down to the left.
        left_added_x = np.arange(left_boundary_x,min_x,x_stepsize) 
        m = 10; b = log_pdf[0] - m*min_x
        left_pdf = m*left_added_x + b 
    
        # Combine the two segments    
        new_x = np.concatenate((left_added_x,x))
        log_pdf = np.concatenate((left_pdf,log_pdf))
        
        entered_left_condition = True
    
    if max_x < right_boundary_x:
        
        # Equation of a line with -10 slope going down to the right.
        right_added_x = np.arange(max_x,right_boundary_x,x_stepsize)
        m = -10; b = log_pdf[-1] - m*max_x
        right_pdf = m*right_added_x + b
        
        # In case new_x is not defined yet if not entered previous condition
        if entered_left_condition:
            new_x = np.concatenate((new_x,right_added_x))
        else:
            new_x = np.concatenate((x,right_added_x))
        log_pdf = np.concatenate((log_pdf,right_pdf))        

    # Normalize the pdf
    log_pdf = np.log10(10**log_pdf/np.linalg.norm(10**log_pdf))
    return new_x, log_pdf



def smooth_cdf_deriv_pdf(x,left_boundary_x,right_boundary_x, plot_path,ion_name,    plotting=False):
    """
    # Only tested on logN. 
    1. Create unbinned CDF
    2. Get only those CDF that is not zero or one. 
    3. Intepolate the CDF
    4. Smooth CDF using savgol_filter
    5. Numerical derivative of CDF -> PDF
    """

    x_stepsize = 0.001 
    sorted_x = np.sort(x); 
    cdf = np.array(range(len(sorted_x)))/float(len(sorted_x))

    inds = np.where((cdf>0) & (cdf<1))[0]    
    sorted_x = sorted_x[inds]; cdf = cdf[inds]
    cdf_intp = interp1d(sorted_x,cdf)
    x_finestep = np.arange(min(sorted_x),max(sorted_x),x_stepsize)
    
    smooth_window_length = int(len(x_finestep) /3.)
    if smooth_window_length % 2 == 0:
        smooth_window_length = smooth_window_length + 1
    cdf_sg_smooth = savgol_filter(cdf_intp(x_finestep),smooth_window_length,3) 
    deriv_pdf     = derivative_cdf(x_finestep,cdf_sg_smooth)
    new_x,new_pdf = truncated_pdf(x_finestep[1:],deriv_pdf,x_stepsize,left_boundary_x,right_boundary_x)
    
    if plotting:
        # Plotting CDF
        pl.figure(1)
        pl.subplots_adjust(hspace=0.2) 
        pl.subplot(211)
        pl.step(sorted_x,cdf,'k',lw=2, label=r'$\rm Sampled\,CDF$')
        pl.plot(x_finestep,cdf_sg_smooth,c='g',lw=2,ls='--', label=r'$\rm Smooth\,CDF$')        
        pl.ylabel(r'$cdf(\log N)$',fontsize=15)
        pl.legend(loc='best')

        pl.subplot(212) # PDF        
        pl.plot(new_x,new_pdf,lw=1,c='m')
        pl.ylim([-4,np.max(new_pdf)+0.1])
        pl.xlim([min(sorted_x),max(sorted_x)]) 
        pl.xlabel(r'$\log N$',fontsize=15)
        pl.ylabel(r'$p(\log N)$',fontsize=15)
        pl.savefig(plot_path + '/pdf_' + ion_name + '.png',bbox_inches='tight',dpi=100)

    return new_x,new_pdf

def write_pdf(x,pdf,output_path,x_name,ion_name):
    """Write to file for the property of the specific ion"""

    fname = output_path + '/' + x_name + '_' + ion_name + '.dat'
    f = open(fname,'w')
    f.write('# %s\t%s\n' % (x_name, 'pdf'))
    for i in xrange(len(x)):
        f.write('%.4f\t%.16f\n' % (x[i], pdf[i]))
    f.close()

def main(config_fname):
    """Extract column density PDF"""
    # Load config parameter object 
    obs_spec = DefineParams(config_fname)
    obs_spec.fileio_mcmc_params()
    obs_spec.fitting_data()
    obs_spec.fitting_params()
    obs_spec.spec_lsf()
    obs_spec.priors_and_init()


    ion_name = raw_input('Ion name = ')
    x_name = 'logN';
    left_boundary_x = 0; right_boundary_x = 22 

    ouput_path = obs_spec.mcmc_outputpath + '/posterior/'
    if not os.path.isdir(ouput_path):
        os.mkdir(ouput_path)
    x = extract_chain(obs_spec.chain_fname,x_name)
    x,pdf = smooth_cdf_deriv_pdf(x,left_boundary_x,right_boundary_x,ouput_path,ion_name,plotting=True)
    write_pdf(x,pdf,ouput_path,x_name,ion_name)


if __name__ == '__main__':
    config_fname = sys.argv[1]
    sys.exit(int(main(config_fname) or 0))
