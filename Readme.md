FyeldGenerator repository
=========================

This package provides with a quick way of generating random field having a specified power spectrum.


Example
-------

```python

from FyeldGenerator import generate_field
import matplotlib.pyplot as plt
import numpy as np

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk

# Draw samples from a normal distribution
def distrib(shape):
    return np.random.normal(size=shape)

shape = (512, 512)

field = generate_field(distrib, Pkgen(2), shape)

plt.imshow(field, cmap='seismic')

```
