
This is a frugal python package with some hyperspectral data goodies.
The whole source, including tests, can be found
[on github](https://github.com/irap-omp/hyperspectral).

How to install
==============

The setup script is build atop `distutils`, so you can read the
[official documentation](https://docs.python.org/2/install/index.html#the-new-standard-distutils)
for installation instructions.

To summarize, you can install via `pip` or via the `setup` script :
```
pip install hyperspectral
```
or
```
python setup.py install
```

How to publish
==============

Copy `.pypirc` to `~/.pypirc`. Fill out the `password` field. (ask me !)

```
python setup.py sdist && twine upload dist/*
```

Dependencies
============

- [astropy](http://www.astropy.org)
- [numpy](http://www.numpy.org)

If you run `setup.py`, you'll need
[distutils](https://docs.python.org/2/install/index.html) too.

Testing Dependencies
====================

You don't need those unless you're running the test suite.

- [unittest](https://docs.python.org/2/library/unittest.html)
- [ddt](https://github.com/txels/ddt)


Hyperspectral Cube
==================

A wrapper for 3D data with two spatial dimensions and one spectral dimension.
It extends `astropy.nddata.NDData` and makes a heavy usage of `astropy.units`.
It main purpose is to sanitize and uniformize hyperspectral cubes from FITS
files that were made by careless astronomers, with header cards (FITS metadata)
ranging from exotic to blatantly illegal.
