import numpy as np
from sksurv.metrics import concordance_index_censored
from wildwood.score import _estimate_concordance_index
import unittest

class Test(unittest.TestCase):
    """A class to test log rank test
    """
    def setUp(self):
        event_indicator = np.array([1, 0, 1, 1, 0, 1]).astype(bool)
        event_time = np.array([3.1, 6.8, 9, 9, 11.3, 16.2])
        estimate = np.array([2.5, 8.2, 8., 10.2, 9.3, 15.2])

        return event_indicator, event_time, estimate
    def test_concordance_index_score(self):
        event_indicator, event_time, estimate = self.setUp()
        val_req = concordance_index_censored(event_indicator, event_time, estimate)[0]
        event_indicator_ = np.ascontiguousarray(event_indicator, dtype=np.bool_)
        event_time_ = np.ascontiguousarray(event_time, dtype=np.float32)
        estimate_ = np.ascontiguousarray(estimate, dtype=np.float32)
        val_test = _estimate_concordance_index(event_indicator_, event_time_, estimate_)
        np.testing.assert_almost_equal(val_test, val_req, decimal=1)

if __name__ == "main":
    unittest.main()