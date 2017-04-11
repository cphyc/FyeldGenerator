# -*- coding: utf-8 -*-

import numpy as np
import six


def generate_field(statistic, power_spectrum, shape, fft=np.fft, fft_args=dict()):
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

    fft: a numpy-like fft API

    fft_args: array
        a dictionary of kwargs to pass to the FFT calls

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
    fftfield = fft.rfftn(field, **fft_args)

    # Compute the k grid
    all_k = [fft.fftfreq(s) for s in shape[:-1]] + \
            [fft.rfftfreq(shape[-1])]
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

        # Prevent overflow by forcing 0 power in k=0 mode
        return np.where(k2 == 0, 0, np.sqrt(power_spectrum(k2)))

    power_k = Pkn(kgrid)
    fftfield *= power_k

    return np.real(fft.irfftn(fftfield, **fft_args))


if __name__ == '__main__':
    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)

        return Pk

    def distrib(shape):
        return np.random.normal(size=shape)
    shape = (512, 512)

    field = generate_field(distrib, Pkgen(2), shape)
