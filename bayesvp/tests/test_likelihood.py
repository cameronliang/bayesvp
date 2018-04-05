import unittest
import os
import sys
import numpy as np

from bayesvp.config import DefineParams
from bayesvp.likelihood import Posterior
from bayesvp.utilities import get_bayesvp_Dir

###############################################################################
# TEST CASE 1: OVI line with stock config file and spectrum
###############################################################################

class TCPosterior(unittest.TestCase):

    def setUp(self):

        # read example config file
        code_path = get_bayesvp_Dir()
        self.config_ex = code_path + '/data/example/config_OVI.dat'
        self.config_params = DefineParams(self.config_ex)
        self.posterior = Posterior(self.config_params)

    def tearDown(self):
        try:
            import shutil
            shutil.rmtree(self.config_params.output_path)
        except OSError as oserr:
            print(oserr)

    ###########################################################################
    # Basic Tests for likelihood, prior and posterior
    ###########################################################################

    def test_default_no_continuum(self):
        self.assertFalse(self.config_params.cont_normalize)

    def test_lnlike(self):
        vp_params = np.array([15,20,0]) # logN, b, z
        correct = -344.55470583729573
        self.assertEqual(self.posterior.lnlike(vp_params),correct)

    def test_prior(self):
        vp_params = np.array([15,20,0])
        correct = 0 
        self.assertEqual(self.posterior.lnprior(vp_params),correct)

        # Outside of prior (logN)
        vp_params = np.array([19,20,0])
        correct = -np.inf
        self.assertEqual(self.posterior.lnprior(vp_params),correct)

        # Outside of prior (b)
        vp_params = np.array([15,-10,0])
        correct = -np.inf
        self.assertEqual(self.posterior.lnprior(vp_params),correct)

        # Outside of prior (z)
        vp_params = np.array([10,20,-1])
        correct = -np.inf
        self.assertEqual(self.posterior.lnprior(vp_params),correct)


    def test_call(self):
        vp_params = np.array([15,20,0])
        correct = -344.55470583729573
        self.assertEqual(self.posterior.__call__(vp_params),correct)

        vp_params = np.array([10,20,-1])
        correct = -np.inf
        self.assertEqual(self.posterior.__call__(vp_params),correct)



if __name__ == '__main__':
    unittest.main()