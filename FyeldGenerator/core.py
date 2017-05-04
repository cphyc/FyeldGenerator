# -*- coding: utf-8 -*-

import numpy as np
import six


def generate_field(statistic, power_spectrum, shape, unit_length=1,
                   fft=np.fft, fft_args=dict()):
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

    unit_length: float
        How much physical length represent 1pixel. For example a value of 10
        mean that each pixel stands for 10 physical units. It has the
        dimension of a physical_unit/pixel.

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

    try:
        fftfreq = fft.fftfreq
        rfftfreq = fft.rfftfreq
    except:
        # Fallback on numpy for the frequencies
        fftfreq = np.fft.fftfreq
        rfftfreq = np.fft.rfftfreq

    # Compute the k grid
    all_k = [fftfreq(s, d=unit_length) for s in shape[:-1]] + \
            [rfftfreq(shape[-1], d=unit_length)]

    kgrid = np.meshgrid(*all_k, indexing='ij')
    knorm = np.sqrt(np.sum(np.power(kgrid), 2), axis=0)

    power_k = np.where(knorm == 0, 0, np.sqrt(power_spectrum(knorm)))
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
