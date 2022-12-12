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
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


shape = (512, 512)

field = generate_field(distrib, Pkgen(2), shape)

plt.imshow(field, cmap="seismic")
```

Install
-------
It is now on pypi!
For the "official" release, use:
```bash
pip install FyeldGenerator
```

For the latest release:
```bash
pip install -e git+https://github.com/cphyc/FyeldGenerator.git#egg=FyeldGenerator
```

License
-------
This work is licensed under the CC-BY-SA license. You are allowed to copy, modify and distribute it as long as you keed the license. See more in the LICENSE file.


Citing
------
If you're using this package for research purposes, consider citing the [Zenodo entry (https://zenodo.org/record/7427712)](https://zenodo.org/record/7427712).