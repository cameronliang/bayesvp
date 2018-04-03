import unittest

from bayesvp.tests import test_config
from bayesvp.tests import test_likelihood
from bayesvp.tests import test_model

suites = []

suites.append(unittest.TestLoader().loadTestsFromTestCase(test_config.TCConfigFile))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_likelihood.TCPosterior))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_model.TCSingleVP))


# Run tests
suite = unittest.TestSuite(suites)
unittest.TextTestRunner(verbosity = 2).run(suite)
