import unittest

from ml_sect.linear_regression.score_model import score_model


class TestLinearRegression(unittest.TestCase):
    def test_model_quality(self):
        self.assertTrue(score_model('insurance.csv') < 15)
