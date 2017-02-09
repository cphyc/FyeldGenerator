# -*- coding: utf-8 -*-

import numpy as np
import six


def generate_field(statistic, power_spectrum, shape):
    """
    Generates a field given a stastitic and a power_spectrum.

    Parameters
    ----------
    statistic: callable
        A function that takes returns a random array of a given signature,
        with signature (s) -> (B) with s == B.shape

    power_spectrum: callable
        A function that returns the power contained in a given mode,
        with signature (k) -> P(k) with k.shape == (ndim, n)

    shape: tuple
        The shape of the output field

    Returns:
    --------
    field: a real array of shape `shape` following the statistic
        with the given power_spectrum
    """

    if not six.callable(statistic):
        raise Exception('`statistic` should be callable')
    if not six.callable(power_spectrum):
        raise Exception('`power_spectrum` should be callable')

    # Draw a random sample
    field = statistic(shape)

    # Compute the FFT of the field
    fftfield = np.fft.rfftn(field)

    # Compute the k grid
    all_k = [np.fft.fftfreq(s) for s in shape[:-1]] + \
            [np.fft.rfftfreq(shape[-1])]
    new_shape = np.array(shape)
    new_shape[-1] = shape[-1] // 2 + 1

    kgrid = [np.zeros(new_shape) for _ in range(len(shape))]

    for i, kg in enumerate(kgrid):
        sl = [slice(None) if j == i
              else None
              for j in range(len(shape))
              ]
        kg[:] = all_k[i][sl]

    def Pkn(kgrid):
        k2 = np.sqrt(np.sum([k**2 for k in kgrid], axis=0))

        @np.vectorize
        def sqrt_0(k2):
            if k2 == 0:
                return 0
            else:
                return np.sqrt(power_spectrum(k2))

        return sqrt_0(k2)

    power_k = Pkn(kgrid)
    fftfield *= power_k

    return np.real(np.fft.irfftn(fftfield))


if __name__ == '__main__':
    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)

        return Pk

    def distrib(shape):
        return np.random.normal(size=shape)
    shape = (512, 512)

    field = generate_field(distrib, Pkgen(2), shape)
