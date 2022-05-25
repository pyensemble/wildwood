import numpy as np
from lifelines.statistics import logrank_test
from wildwood._log_rank_test import compute_test_statistic
import unittest

class Test(unittest.TestCase):
    """A class to test log rank test
    """
    def setUp(self):
        t1 = np.array([3.1, 6.8, 9, 9, 11.3, 16.2])
        d1 = np.array([1, 0, 1, 1, 0, 1])
        t2 = np.array([8.7, 9, 10.1, 12.1, 18.7, 23.1])
        d2 = np.array([1, 1, 0, 0, 1, 0])
        return t1, d1, t2, d2

    def test_several_log_rank(self):
        t1, d1, t2, d2 = self.setUp()
        val_req = logrank_test(t1, t2, event_observed_A=d1, event_observed_B=d2).test_statistic
        t1_ = np.ascontiguousarray(t1, dtype=np.float32)
        t2_ = np.ascontiguousarray(t2, dtype=np.float32)
        d1_ = np.ascontiguousarray(d1, dtype=np.int32)
        d2_ = np.ascontiguousarray(d2, dtype=np.int32)
        val_test = compute_test_statistic(t1_, t2_, d1_, d2_)**2
        np.testing.assert_almost_equal(val_test, val_req, decimal=1)

if __name__ == "main":
    unittest.main()