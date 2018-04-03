import unittest
import os
import sys
import numpy as np


from bayesvp.vp_model import general_intensity
from bayesvp.vp_model import wavelength_array
from bayesvp.vp_model import simple_spec

###############################################################################
# TEST CASE 1: OVI line with stock config file and spectrum
###############################################################################

class TCSingleVP(unittest.TestCase):

    def setUp(self):
        # OVI line
        self.logN = 14.5; self.b = 20; self.z = 0
        self.fosc = 1.32500000e-01; self.wave0 = 1.03192610e+03
        self.gamma = 4.14900000e+08; self.mass_oxy = 2.65676264e-23
        self.atomic_params = np.array([self.fosc,self.wave0,self.gamma,self.mass_oxy])

    def test_general_intensity(self):

        norm_intensity = general_intensity(self.logN, self.b, self.z, 
                        self.wave0, self.atomic_params)
        self.assertAlmostEqual(norm_intensity,0.040468,places=5)


    def test_simple_spec(self):
        wave = wavelength_array(1031, 1033, 7.5)
        flux = simple_spec(self.logN,self.b,self.z,wave,'O','VI')
        self.assertEqual(np.shape(wave),(77,))
        self.assertEqual(np.shape(flux),np.shape(wave))


if __name__ == '__main__':

    unittest.main()