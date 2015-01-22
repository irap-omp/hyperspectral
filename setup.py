#!/usr/bin/env python

# Run this script to leverage the power of `distutils` to install or build.

# Doc :
# - https://docs.python.org/2/install/index.html
# - https://docs.python.org/2/distutils/setupscript.html

from distutils.core import setup

with open('./VERSION', 'r') as version_file:
    __version__ = version_file.read().strip()

setup(
    name='hyperspectral',
    version=__version__,
    author='Goutte',
    author_email='antoine@goutenoir.com',
    maintainer='Goutte',
    maintainer_email='antoine@goutenoir.com',
    url='https://github.com/irap-omp/hyperspectral',
    description="A frugal python package with some hyperspectral data goodies.",
    license='Science!',

    package_dir={'hyperspectral': 'lib'},
    packages=['hyperspectral'],
    data_files=[('', [
        'README.md',
        'VERSION',
    ])],

    requires=['astropy', 'numpy'],
    provides=['hyperspectral'],
)