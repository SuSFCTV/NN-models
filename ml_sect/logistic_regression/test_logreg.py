import unittest

from ml_sect.logistic_regression.score_model import score_model


class TestLinearRegression(unittest.TestCase):
    def test_model_quality(self):
        self.assertTrue(score_model('water_potability.csv', ['ph', 'Sulfate', 'Trihalomethanes']) > 0.35)
