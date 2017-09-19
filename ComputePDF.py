################################################################################
#
# ComputePDF.py 		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Extract the one dimensional marginalized probability density distribution 
# for property x (e.g., x = logN).    
################################################################################

import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import sys
import os

from Config import DefineParams
from Utilities import compute_burnin_GR

def extract_chain(obs_spec,para_name):
    """
    This assumes simple single component model
    with parameter [logN, b, z]
    """

    #print 'vp_params_type = ', obs_spec.vp_params_type
    chain = np.load(obs_spec.chain_fname + '.npy')
    my_dict = {'logN':0, 'b':1,'z':2}
    col_num = my_dict[para_name]
    burnin = compute_burnin_GR(obs_spec.chain_fname + '_GR.dat')
    
    # return 1D chain of parameter x 
    return chain[burnin:,:,col_num].flatten()

def spline_binned_pdf(x,bins = 30, interp_kind = 'linear'):
    """
    Convert from MCMC samples to PDF by interpolating 
    the histogram
    """
    pdf, edges = np.histogram(x,density=1,bins=bins)
    f = interp1d(edges[1:],pdf,kind=interp_kind)

    new_x = np.arange(min(edges[1:]),max(edges[1:]),0.001)

    return new_x, f(new_x) 

def extrapolate_pdf(x,pdf,left_boundary_x,right_boundary_x,slope=10):
    """ 
    Extrapolate the log10(pdf) outside the range of (min_x,max_x) with 
    some logarithmic slope 
    """

    log_pdf = np.log10(pdf)
    min_x = min(x); max_x = max(x);
    x_stepsize = np.median(x[1:]-x[:-1])
    
    entered_left_condition = False
    if min_x > left_boundary_x:

        # equation of a line with +10 slope going down to the left.
        left_added_x = np.arange(left_boundary_x,min_x,x_stepsize) 
        m = slope; b = log_pdf[0] - m*min_x
        left_pdf = m*left_added_x + b 
    
        # Combine the two segments    
        new_x = np.concatenate((left_added_x,x))
        log_pdf = np.concatenate((left_pdf,log_pdf))
        
        entered_left_condition = True
    
    if max_x < right_boundary_x:
        
        # Equation of a line with -10 slope going down to the right.
        right_added_x = np.arange(max_x,right_boundary_x,x_stepsize)
        m = -slope; b = log_pdf[-1] - m*max_x
        right_pdf = m*right_added_x + b
        
        # In case new_x is not defined yet if not entered previous condition
        if entered_left_condition:
            new_x = np.concatenate((new_x,right_added_x))
        else:
            new_x = np.concatenate((x,right_added_x))
        log_pdf = np.concatenate((log_pdf,right_pdf))        

    # Normalize the pdf
    log_pdf = np.log10(10**log_pdf/np.sum((10**log_pdf)*(x_stepsize)))
    return new_x, log_pdf

def plot_pdf(x,pdf,plot_path,ion_name):
    """Plot the distribution of property x"""
    
    pl.figure(1)        
    pl.plot(x,pdf,lw=2,c='m')
    pl.xlim([min(x),max(x)]) 
    pl.xlabel(r'$\log N$',fontsize=15)
    pl.ylabel(r'$p(\log N)$',fontsize=15)
    pl.savefig(plot_path + '/pdf_' + ion_name + '.png',bbox_inches='tight',dpi=100)
    pl.clf()

def write_pdf(x,pdf,output_path,x_name,ion_name):
    """Write to file for the distribution of the specific ion"""

    fname = output_path + '/' + x_name + '_' + ion_name + '.dat'
    f = open(fname,'w')
    f.write('# %s\t%s\n' % (x_name, 'pdf'))
    for i in xrange(len(x)):
        f.write('%.4f\t%.16f\n' % (x[i], pdf[i]))
    f.close()

    print('Written %s' % fname)

def main(config_fname,ion_name,nbins,interp_kind):
    """Extract PDF of property x (logN, b, or z)"""
    # Load config parameter object 
    obs_spec = DefineParams(config_fname)

    x_name = 'logN';
    left_boundary_x = 0; right_boundary_x = 22 

    ouput_path = obs_spec.mcmc_outputpath + '/posterior'
    if not os.path.isdir(ouput_path):
        os.mkdir(ouput_path)

    x = extract_chain(obs_spec,x_name)
    linear_x,linear_pdf = spline_binned_pdf(x,nbins,interp_kind)
    x,logpdf = extrapolate_pdf(linear_x,linear_pdf,left_boundary_x,right_boundary_x)

    plot_pdf(linear_x,linear_pdf,ouput_path,ion_name)
    write_pdf(x,logpdf,ouput_path,x_name,ion_name)


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print('python ComputePDF.py full_path_config_filename output_name nbins interpolation_order')
        print('nbins = number of bins in histogram of PDF')
        print('interpolation_order = linear or cubic, etc see scipy.interpolate.interp1d\n')
        
        exit()
    else:
        config_fname = sys.argv[1]
        ion_name     = sys.argv[2]
        nbins        = int(sys.argv[3])
        interp_kind  = sys.argv[4]
        sys.exit(int(main(config_fname,ion_name,nbins,interp_kind) or 0))
