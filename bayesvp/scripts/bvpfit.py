################################################################################
#
# bvpfit.py         (c) Cameron Liang
#                   University of Chicago
#                   jwliang@oddjob.uchicago.edu
#
# Main program to execute the fitting process.
#
################################################################################

import sys
import os

from bayesvp.mcmc_setup import bvp_mcmc
from bayesvp.utilities import MyParser

def main(config_fname = None):
    parser = MyParser()
    parser.add_argument('config_fname',help="full path to config filename", nargs='?')
    parser.add_argument("-t", "--test",help="a mcmc fitting test with default config and spectrum",
                        action="store_true")
    parser.add_argument("-pc", "--printconfig",help="print config parameters to screen",
                        action="store_true")
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.test:
        from bayesvp.utilities import get_bayesvp_Dir
        path = get_bayesvp_Dir()
        config_fname = path + '/data/example/config_OVI.dat'
        args.printconfig = True
        bvp_mcmc(config_fname,args.printconfig)

    if args.config_fname: 
        if os.path.isfile(args.config_fname):
            bvp_mcmc(args.config_fname,args.printconfig)
        else:
            sys.exit('Config file does not exist:\n %s' % args.config_fname)

if __name__ == '__main__':
    sys.exit(main() or 0)
