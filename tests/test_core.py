# -*- coding: utf-8 -*-
import unittest
from FyeldGenerator import generate_field
import numpy as np


class TestFieldGenerator(unittest.TestCase):

    def setUp(self):
        shape = 128, 128

        def spectrum(n):
            def Pk(k):
                return np.power(k, -n)
            return Pk

        def statistic(shape):
            return 1 * np.exp(np.random.rand(*shape) * 2*np.pi)

        self.field = generate_field(statistic=statistic,
                                    power_spectrum=spectrum(2),
                                    shape=shape)

    def test_histogram(self):
        pass
