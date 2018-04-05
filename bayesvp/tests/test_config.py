import unittest
import os
import sys
import numpy as np

from bayesvp.config import DefineParams
from bayesvp.utilities import get_bayesvp_Dir

"""
TEST CASE 1: OVI line with stock config file and spectrum


The stock config file at bayesvp/data/example/bvp_configs/config_OVI.dat
should contain the following:

spec_path test_path_to_spec
output o6
mcmc 100 200 4 bic kombine
%% OVI.spec 1030.000000 1033.000000
% O VI 15 30 0.000000
logN 10.00 18.00
b    0.00 100.00
z    0.000000 300.00
"""

class TCConfigFile(unittest.TestCase):

    def setUp(self):
        code_path = get_bayesvp_Dir()
        # read example config file
        self.config_ex = code_path + '/data/example/config_OVI.dat'
        self.config_params = DefineParams(self.config_ex)

    def tearDown(self):
        try:
            import shutil
            shutil.rmtree(self.config_params.output_path)
        except OSError as oserr:
            print(oserr)

    def test_config_file_exists(self):
        self.assertTrue(os.path.isfile(self.config_ex))

    def test_default_no_continuum_params(self):
        self.assertFalse(self.config_params.cont_normalize)
        self.assertEqual(self.config_params.cont_nparams,0)


    def test_spectral_data(self):

        self.assertTrue(os.path.isfile(self.config_params.spec_fname))

        self.assertEqual(len(self.config_params.flux),len(self.config_params.dflux))
        self.assertEqual(len(self.config_params.wave),len(self.config_params.dflux))

        self.assertFalse(np.isnan(self.config_params.wave).any())
        self.assertFalse(np.isnan(self.config_params.flux).any())
        self.assertFalse(np.isnan(self.config_params.dflux).any())



    def test_example_mcmc_params(self):
        self.assertEqual(self.config_params.mcmc_sampler,'kombine')
        self.assertEqual(self.config_params.model_selection,'bic')
        
        self.assertEqual(self.config_params.n_component,1)
        self.assertEqual(self.config_params.n_params,self.config_params.n_component*3)
        
        self.assertEqual(self.config_params.nsteps,200)
        self.assertEqual(self.config_params.nthreads,4)
        self.assertEqual(self.config_params.nwalkers,100)

    def test_example_priors(self):
        self.assertEqual(np.shape(self.config_params.priors),(3,2))

        self.assertAlmostEqual(self.config_params.priors[0][0],10.0)
        self.assertAlmostEqual(self.config_params.priors[0][1],18.0)

        self.assertAlmostEqual(self.config_params.priors[1][0],0.0)
        self.assertAlmostEqual(self.config_params.priors[1][1],100.0)

        self.assertAlmostEqual(self.config_params.priors[2][0],-0.00100069)
        self.assertAlmostEqual(self.config_params.priors[2][1],0.00100069)

        self.assertEqual(float(self.config_params.redshift),0.0)

    def test_transistion_params_array(self):
        self.assertFalse(np.isnan(self.config_params.transitions_params_array).any())
        self.assertEqual(np.shape(self.config_params.transitions_params_array),(1,1,1,4))
        self.assertIsNotNone(self.config_params.transitions_params_array)
        self.assertFalse(np.isnan(self.config_params.transitions_params_array[0][0][0][1]))

    def test_vp_params(self):
        self.assertEqual(self.config_params.wave_begins,1030.0)
        self.assertEqual(self.config_params.wave_ends,1033.0)

        self.assertEqual(self.config_params.vp_params_flags[0],0)
        self.assertEqual(self.config_params.vp_params_flags[1],1)
        self.assertEqual(self.config_params.vp_params_flags[2],2)

        self.assertEqual(self.config_params.vp_params_type[0],'logN')
        self.assertEqual(self.config_params.vp_params_type[1],'b')
        self.assertEqual(self.config_params.vp_params_type[2],'z')

        self.assertEqual(self.config_params.vp_params[0][0],'15')
        self.assertEqual(self.config_params.vp_params[0][1],'30')
        self.assertEqual(self.config_params.vp_params[0][2],'0.000000')


if __name__ == '__main__':
    unittest.main()