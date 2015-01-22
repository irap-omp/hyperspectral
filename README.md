
This is a frugal python package of miscellaneous hyperspectral data goodies.

Dependencies
============

- [astropy](http://www.astropy.org)
- [numpy](http://www.numpy.org)


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
