import numpy as np


def generate_field(
    statistic,
    power_spectrum,
    shape,
    unit_length=1,
    fft=np.fft,
    fft_args=None,
    stat_real=False,
):
    """
    Generates a field given a stastitic and a power_spectrum.

    Parameters
    ----------
    statistic: callable
        A function that takes returns a random array of a given signature,
        with signature (s) -> (B) with B.shape == s. Please note that the
        distribution is in *Fourier space* not in real space, unless you set
        stat_real=True. See the note below for more details.

    power_spectrum: callable
        A function that returns the power contained in a given mode,
        with signature (k) -> P(k) with k.shape == (ndim, n)

    shape: tuple
        The shape of the output field

    unit_length: float, optional
        How much physical length represent 1pixel. For example a value of 10
        mean that each pixel stands for 10 physical units. It has the
        dimension of a physical_unit/pixel.

    fft: a numpy-like fft API, optional

    fft_args: array, optional
        a dictionary of kwargs to pass to the FFT calls

    stat_real: boolean, optional
        Set to true if you want the distribution to be drawn in real space and
        then transformed into Fourier space.

    Returns
    -------
    field: a real array of shape `shape` following the statistic
        with the given power_spectrum

    Note
    ----
    When generation the distribution in Fourier mode, the result
    should be complex and unitary. Only the phase is random.
    """

    if not callable(statistic):
        raise Exception("`statistic` should be callable")
    if not callable(power_spectrum):
        raise Exception("`power_spectrum` should be callable")

    try:
        fftfreq = fft.fftfreq
    except NameError:
        # Fallback on numpy for the frequencies
        fftfreq = np.fft.fftfreq

    # Compute the k grid
    all_k = [fftfreq(s, d=unit_length) for s in shape]

    kgrid = np.meshgrid(*all_k, indexing="ij")
    knorm = np.sqrt(np.sum(np.power(kgrid, 2), axis=0))

    fourier_shape = knorm.shape

    if fft_args is None:
        fft_args = {}

    if stat_real:
        field = statistic(shape)
        # Compute the FFT of the field
        fftfield = fft.fftn(field, **fft_args)
    else:
        # Draw a random sample in Fourier space
        fftfield = statistic(fourier_shape)

    power_k = np.zeros_like(knorm)
    mask = knorm > 0
    power_k[mask] = np.sqrt(power_spectrum(knorm[mask]))
    fftfield *= power_k

    return np.real(fft.ifftn(fftfield, **fft_args))


if __name__ == "__main__":

    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)

        return Pk

    def distrib(shape):
        # Build a unit-distribution of complex numbers with random phase
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    shape = (512, 512)

    field = generate_field(distrib, Pkgen(2), shape)
