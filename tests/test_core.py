import unittest

import numpy as np

from FyeldGenerator import generate_field


class TestFieldGenerator(unittest.TestCase):
    def setUp(self):
        shape = 128, 128

        def spectrum(n):
            def Pk(k):
                return np.power(k, -n)

            return Pk

        def statistic(shape):
            a = np.random.normal(loc=0, scale=1, size=shape)
            b = np.random.normal(loc=0, scale=1, size=shape)
            return a + 1j * b

        self.field = generate_field(
            statistic=statistic, power_spectrum=spectrum(2), shape=shape
        )

    def test_histogram(self):
        pass
