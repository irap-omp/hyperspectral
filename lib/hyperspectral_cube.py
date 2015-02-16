# coding=utf-8

from __future__ import division

import logging
from copy import copy
import numpy as np

from astropy.nddata import NDData
from astropy.units import Unit, Quantity
from astropy.io import fits

from axes import Axis, UndefinedAxis

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('HyperspectralCube')


# Some todos
#
#Sanitization operations :
#    - IDL-made FITS :
#        - keyword FSCALE (http://astro.uni-tuebingen.de/software/idl/astrolib/fits/mrdfits.html)
#        - other non-ISO header blunders -_-
#
#Use SI units, http://www.aanda.org/articles/aa/full_html/2010/16/aa15362-10/aa15362-10.html#S39
#Uniformization operations :
#    - Spatial Units !
#    - Spectral Units !


class HyperspectralCube(NDData):
    """
    A hyperspectral cube has 3 dimensions : 2 spatial, and 1 spectral.
    It is basically a rectangular region of the sky seen on a discrete subset of
    the light spectrum. It is heavily used in spectral imaging, and is provided
    by spectroscopic explorers such as MUSE.

    This class is essentially a convenience wrapper for data and metadata of a
    single hyperspectral cube, built atop `astropy`'s growing `NDData`. As such,
    it may be subject to breaking changes without notice. The API will be deemed
    frozen and stable when astropy will be released as v1.0.0.

    It understands indexation (getting and setting) of the form
    ``cube[zmin:zmax, ymin:ymax, xmin:xmax]``, like numpy's ndarray would.

    Throughout the code and documentation, `z` is sometimes referred as `λ`,
    because the Z axis is the spectral one, and `λ` is often used as a symbol to
    describe a wavelength.

    Notes
    -----

    - No FSCALE management is implemented yet.
    - A ton of other metadatas are not supported yet.
      Add support for what you need !

    Factories
    ---------

    You can load from a single HDU of a FITS file, if you do not provide one it
    will try to detect HDUs with data and pick the first one: ::

        cube = HyperspectralCube.from_fits(filepath, hdu_index=None)

    Parameters
    ----------

    data: numpy.ndarray|None
        3D numpy ndarray holding the raw data.
        The indices are the pixel coordinates in the cube.
        Note that in order to be consistent with astropy.io.fits,
        data is indexed in that order : λ, y, x.

    verbose: boolean
        Set to True to log everything and anything.

    ... and other parameters inherited from NDData.

    """

    def __init__(self, data=None, x=None, y=None, z=None,
                 uncertainty=None, mask=None, flags=None,
                 wcs=None, meta=None, unit=None, verbose=False):
        super(HyperspectralCube, self).__init__(
            data, uncertainty=uncertainty,
            mask=mask, flags=flags,
            wcs=wcs, meta=meta, unit=unit
        )
        self.verbose = verbose
        if x is None:
            x = UndefinedAxis('x')
        if y is None:
            y = UndefinedAxis('y')
        if z is None:
            z = UndefinedAxis('z')
        self.x = x
        self.y = y
        self.z = z
        self.axes = [z, y, x]

    ## FACTORIES ###############################################################

    @staticmethod
    def from_fits(filename, hdu_index=None, verbose=False):
        """
        Factory to create a HyperspectralCube from one HDU in a FITS file.
        http://fits.gsfc.nasa.gov/fits_standard.html
        http://idlastro.gsfc.nasa.gov/ftp/pro/astrom/aaareadme.txt

        You may specify the index of the HDU you want. If you don't, it will try
        to guess by searching for a HDU with EXTNAME=DATA, or it will pick the
        first HDU in the list that has 3 dimensions.

        This is the heart of this class, where the nitty-gritty code is written.

        filename: string
            An absolute or relative path to the FITS file we want to load.
            The `astropy.io.fits` module is used to open the file.
        hdu_index: integer|None
            Index of the HDU to load.
            If you set this, guessing is not attempted.
        verbose: boolean
            Should we fill up the log with endless chatter ?

        :rtype: HyperspectralCube
        """
        try:
            hdu_list = fits.open(filename)
        except Exception:
            raise IOError("Could not open provided FITS file '%s'" % filename)

        if verbose:
            log.info("Opening FITS file %s.", filename)

        # I. Find out which HDU we should read the data from
        hdu = header = None

        def _sanity_check_dimensions(_header, _hdu_index):
            if _header.get('NAXIS') != 3:
                raise ValueError("Wrong dimensions : NAXIS of HDU #%d is not 3",
                                 _hdu_index)

        # 1/ If set, use HDU indexed at hdu_index
        if hdu_index is not None:
            if hdu_index < 0 or hdu_index >= len(hdu_list):
                raise ValueError("HDU #%d is not available", hdu_index)
            hdu = hdu_list[hdu_index]
            header = hdu.header
            _sanity_check_dimensions(header, hdu_index)
        else:
            # 2/ Try to find DATA extension HDU
            found_data_hdu = False
            hdu_index = 0
            while not found_data_hdu and hdu_index < len(hdu_list):
                hdu = hdu_list[hdu_index]
                header = hdu.header
                if 'data' in header.get('EXTNAME', '').lower():
                    try:
                        _sanity_check_dimensions(header, hdu_index)
                    except ValueError:
                        hdu_index += 1
                    else:
                        found_data_hdu = True
                else:
                    hdu_index += 1

            # 3/ Try any other HDU, starting with Primary
            if not found_data_hdu:
                hdu_index = 0
                while not found_data_hdu and hdu_index < len(hdu_list):
                    hdu = hdu_list[hdu_index]
                    header = hdu.header
                    try:
                        _sanity_check_dimensions(header, hdu_index)
                    except ValueError:
                        hdu_index += 1
                    else:
                        found_data_hdu = True

            # 4/ Alert! Exception raised
            if not found_data_hdu:
                raise ValueError("Could not find a HDU containing 3D data.")

        # II. Collect and numpyfy data
        data = np.array(hdu.data, dtype=float)

        def _get_meta(_header, _names, default=None):
            """
            Local utility to fetch a header card value that can have many names.
            Will try (in order) the names in _names and stop at first found.
            If none is found and the default is not set, will yell.
            """
            _meta = None
            k = 0
            while _meta is None and k < len(_names):
                _meta = _header.get(_names[k])
                k += 1
            if _meta is None:
                _meta = default
            if _meta is None:
                headerdump = _header.tostring("\n        ").strip()
                raise IOError("Did not find any header card for any of (%s) "
                              "in header:\n%s" % (",".join(_names), headerdump))
            return _meta

        def _get_axis_from_header(_header, _id):
            """
            Local utility to create an Axis object from the data we can extract
            from the FITS header. As FITS files authors are pretty whimsical
            regarding the names of the header cards, we check for as many names
            as we can.

            Note that _id
            """
            _step = _get_meta(_header, [
                'CDELT%d' % _id,
                'CD%d_%d' % (_id, _id),
                'CDEL_%d' % _id,
            ])
            # Make sure our start value is for pixel 0, not CRPIX.
            # Note that the 1st pixel has a CRPIX value of 1, not 0.
            _val = _get_meta(_header, ['CRVAL%d' % _id])
            _start = _val - (_get_meta(_header, ['CRPIX%d' % _id]) - 1.) * _step
            return Axis(
                _get_meta(_header, ['CTYPE%d' % _id], default='Axis%d' % _id),
                _start,
                _step,
                Unit(_get_meta(_header, ['CUNIT%d' % _id])),
            )

        # III. Collect and sanitize metadata
        # http://docs.astropy.org/en/latest/io/fits/usage/verification.html
        hdu.verify(option='fix')
        HyperspectralCube.sanitize_fits_header(header)
        x = _get_axis_from_header(header, 1)
        y = _get_axis_from_header(header, 2)
        z = _get_axis_from_header(header, 3)

        meta = {
            'fits': header
        }

        return HyperspectralCube(data=data, x=x, y=y, z=z,
                                 meta=meta, verbose=verbose)

    # @staticmethod
    # def from_mpdaf(cube, verbose=False):
    #     """
    #     Factory to create a cube from a `MPDAF Cube <http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib/user_manual_cube.html>`_.
    #     """
    #     header = cube.data_header
    #     data = cube.data
    #
    #     return HyperspectralCube(data=data, header=header, verbose=verbose)

    ## STATICS #################################################################

    @staticmethod
    def sanitize_fits_header(header):
        """
        Sanitizes provided FITS header, by procedurally applying various
        sanitization tasks :

        - Fix step keyword for each axis:
            - CDEL_3  --> CDELT3
            - CD3_3   --> CDELT3
        - Fix blatantly illegal units :
            - DEG     --> deg
            - MICRON  --> um
            - MICRONS --> um

        This is where we'll add additional sanitization tasks as users report
        failing cubes.

        .. warning::
            The header is mutated, not copied, so this method returns nothing.
        """
        if header is None:
            return

        # Fix step keyword on each axis
        for i in range(1, 4):
            if header.get('CDELT%d' % i) is None:
                if header.get('CDEL_%d' % i) is not None:
                    header.set('CDELT%d' % i, header.get('CDEL_%d' % i))
                if header.get('CD%d_%d' % (i, i)) is not None:
                    header.set('CDELT%d' % i, header.get('CD%d_%d' % (i, i)))

        # Fix illegal units on each axis
        for i in range(1, 4):
            unit_keyword = 'CUNIT%d' % i
            unit = header.get(unit_keyword)
            if unit is not None:
                unit = unit.strip()
                if unit == 'DEG':
                    unit = 'deg'
                elif 'micron' in unit.lower():
                    unit = 'um'
                header.set(unit_keyword, unit)

    ## API - EXPORT ############################################################

    def to_fits(self, filepath, clobber=False):
        """
        Write this cube to a FITS file located at `filepath`, which *must* end
        with the extension `.fits`.

        filepath: string
            The filepath (absolute or relative) of the file we want to write to.
            The `astropy.io.fits` module is used to write to the file.
            Examples : 'out/my_cube.fits' or '/var/fits/my_cube.fits'.
        clobber: bool
            When set to True, will overwrite the output file if it exists.
        """
        if 'fits' in self.meta:
            header = self.meta['fits']
        else:
            header = fits.Header()
            header['CDELT3'] = self.axes[0].step
            # todo: fill up the header with information from the axes
            raise NotImplementedError(
                "Cannot write a cube to a FITS file if it has not been created "
                "from a FITS file. Ask for the feature if you need it !"
            )
        primary_hdu = fits.PrimaryHDU(data=self.data, header=header)
        hdulist = fits.HDUList([primary_hdu])
        hdulist.writeto(filepath, clobber=clobber)
        if self.verbose:
            log.info("Writing HyperspectralCube to file %s.", filepath)
            # Note: astropy already logs something similar

    ## API - INFORMATION #######################################################

    def is_empty(self):
        """
        Is this cube void of any data ?

        return boolean
        """
        return self.data is None or len(self.data.shape) == 0

    def copy(self, out=None):
        """
        Copies this cube into `out` (if specified) and returns the copy.

        Not sure about the usefulness of the `out` parameter ; may change later.

        :rtype: HyperspectralCube
        """
        data = copy(self.data)

        out = HyperspectralCube(
            data=data,
            x=self.x.copy(), y=self.y.copy(), z=self.z.copy(),
            uncertainty=copy(self.uncertainty), wcs=copy(self.wcs),
            mask=copy(self.mask), flags=copy(self.flags),
            meta=copy(self.meta), unit=copy(self.unit)
        )

        return out

    def has_metadata(self, axis=None):
        if axis is None:
            return \
                self.has_metadata(0) and \
                self.has_metadata(1) and \
                self.has_metadata(2)
        else:
            if axis < 0 or axis > 2:
                raise ValueError("Invalid axis '%s'. "
                                 "Accepted values: 0,1,2." % axis)
            # if it's not the default UndefinedAxis, we're ok.
            return not isinstance(self.axes[axis], UndefinedAxis)

    def get_step(self, axis):
        """
        Returns the step along `axis` in the unit specified by the header.
        Axis values are following FITS conventions:
            - 0 for λ
            - 1 for y
            - 2 for x

        :rtype: astropy.units.Quantity
        """
        if not self.has_metadata(axis):
            raise ValueError("Cannot get the step along axis #%s "
                             "of a cube without metadata." % axis)

        return Quantity(self.axes[axis].step, self.axes[axis].unit)

    def get_steps(self):
        """
        Returns a list of the 3 steps for the axes [λ,y,x], in that order,
        as `astropy.units.Quantity`.

        :rtype: list of astropy.units.Quantity
        """
        if not self.has_metadata():
            raise IOError("Cannot get the steps of a cube without metadata.")
        return [self.get_step(0), self.get_step(1), self.get_step(2)]

    def pixel_of(self, wavelength):
        """
        Returns the pixel index (starting at 0) for the passed `wavelength`,
        whose unit is assumed to be the one specified in the metadata for the
        spectral axis. You can provide a Quantity object with its own Unit, as
        long as it is compatible with the Unit of the Z axis. For instance,
        centimeters and Angstroms are compatible, but Hertz and meters are not.

        The parameter `wavelength` may also be a list of wavelengths.

        :rtype: float | list of float
        """
        if isinstance(wavelength, str):
            wavelength = float(wavelength)
        if not isinstance(wavelength, Quantity):
            wavelength = Quantity(wavelength, self.z.unit)

        # Creating a Quantity object ensures that the units are merged properly,
        # because sometimes they are not and we get a (cm) / (m) Unit
        return Quantity((wavelength - self.z.start) / self.z.step, '').value

    def wavelength_of(self, pixel):
        """
        Get the wavelength (in the unit specified in the header) of the
        specified pixel index along z.
        The pixel index should start at 0.

        The parameter `pixel` may also be a list of pixel indices.

        :rtype: Quantity
        """
        if isinstance(pixel, str):
            pixel = float(pixel)
        if not isinstance(pixel, Quantity):
            pixel = Quantity(pixel, '')

        return pixel * self.z.step + self.z.start

    ## MAGIC METHODS ###########################################################

    def __str__(self):
        """
        Simple but useful string representation of the metadata and the data.
        """
        if self.is_empty():
            data = "None"
        else:
            data = "ndarray of shape %s" % str(self.data.shape)

        meta = ''
        indent = '        '
        for key in self.meta:
            print key + 'caca'
            meta += '\n' + indent + key
            if key == 'fits' and isinstance(self.meta[key], fits.Header):
                meta += ':\n' + indent+indent + \
                        self.meta[key].tostring("\n"+indent+indent).strip()
            else:
                meta += ': %s' % self.meta[key]

        return """HyperspectralCube
    meta : {meta}
    data : {data}
    z: {s.z}
    y: {s.y}
    x: {s.x}
""".format(s=self, meta=meta, data=data)

    def __getitem__(self, key):
        """
        Usage example :
        cube[λmin:λmax:λstep, ymin:ymax:ystep, xmin:xmax:xstep]

        This selector is quite similar to python's own slicer : min is included,
        max is excluded, and the default step is 1.
        Here's a reference documentation about slicing :
        http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

        .. note ::
            As we're using numpy's ndarrays for the data, internally, slicing
            creates *views* (shallow copies) of the data.
            See http://docs.scipy.org/doc/numpy/glossary.html#term-view

        .. warn ::
            Negative steps are not supported at the moment. Feel free to
            implement support for them ! There's a test suite for that ;)

        :rtype: HyperspectralCube
        """
        if self.is_empty():
            raise KeyError("Cannot use [:,:,:] selector on empty cube.")

        # We need the shape before the cut, to handle negative values in `key`.
        old_shape = self.shape

        # Make the cut using parent's method
        new_data = super(HyperspectralCube, self).__getitem__(key)

        # Utility function to create an axis that is a slice of another
        def _get_adjusted_axis(_axis, _index, _key):
            start = _axis.start
            start_index = _key[_index].start
            if not (start_index is None or start_index == 0):
                if start_index < 0:
                    start_index = old_shape[_index] + start_index
                start = start + start_index * _axis.step

            step = _axis.step
            step_index = _key[_index].step
            if not (step_index is None or step_index == 1):
                if step_index < 0:
                    # Oh, this is more complex than it seems, as it affects
                    # start and stop values, too.
                    # We'll postpone it for now ; add a test case and hack away!
                    raise NotImplementedError(
                        "Negative steps are not supported at the moment. "
                        "Make a request or add support for them yourself!"
                    )
                step = step * step_index

            return Axis(_axis.name, start, step, _axis.unit)

        # Make sure that our key is a 3-tuple, by creating bogus missing slices
        # Eg: cube[0:1, :, 0:-1] yields in `key` the following value :
        # (slice(0, 1, None), slice(None, None, None), slice(None, -1, None))
        # But: cube[0:2] yields only slice(0, 2, None) (not even a tuple)
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1:
            key = (key[0], slice(0), slice(0))
        elif len(key) == 2:
            key = (key[0], key[1], slice(0))
        elif len(key) == 3:
            pass  # 'cause everything is awesome!
        else:
            raise TypeError("Selector %s has too many elements." % key)

        # Slice the metadata : the axes
        new_data.z = _get_adjusted_axis(self.z, 0, key)
        new_data.y = _get_adjusted_axis(self.y, 1, key)
        new_data.x = _get_adjusted_axis(self.x, 2, key)
        new_data.axes = [new_data.z, new_data.y, new_data.x]

        return new_data

    def __setitem__(self, key, value):
        """
        cube[λmin:λmax, ymin:ymax, xmin:xmax] = Number
        cube[λmin:λmax, ymin:ymax, xmin:xmax] = numpy.ndarray

        This mutates this cube's data and does not return anything.

        The indexed mutator is quite similar to numpy's ndarray's :
        min is included, max is excluded.
        You can use negative values, Python slicing syntax, as described here :
        http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

        value: numpy.ndarray or Number
            Must be broadcastable to shape defined by the [λ,y,x] slice.

        """
        if self.is_empty():
            raise KeyError("Cannot use [:,:,:] setter on empty cube.")

        self.data[key] = value  # numpy will raise appropriate errors

    ## ARITHMETIC ##############################################################

    def __add__(self, other):
        """
        Addition should work pretty much like numpy's ndarray addition.

        HyperspectralCube + Number = HyperspectralCube
            => add the number to each voxel of the cube.

        HyperspectralCube + ndarray = HyperspectralCube
            => requires the ndarray to be of broadcast-compatible shape.

        ndarray + HyperspectralCube = ndarray
            => requires the ndarray to be of broadcast-compatible shape.
               The result is a ndarray. This can't be helped ; deal with it.

        HyperspectralCube + HyperspectralCube = HyperspectralCube
            => requires the cubes to be of same shapes and same referentials.
               This operation could be improved to accept cubes of different
               but compatible referentials.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot add to an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot add an empty HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot add HyperspectralCubes with different "
                                "referentials.")
            result = self.add(other)
            result.x = self.x
            result.y = self.y
            result.z = self.z
            return result
        else:
            # other types, numpy will raise when inappropriate
            data = self.data + other

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __radd__(self, other):
        """
        Addition following Peano axioms is commutative on our sets.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction should work pretty much like numpy's ndarray subtraction.

        HyperspectralCube - Number = HyperspectralCube
            => subtract the number to each voxel of the cube.

        HyperspectralCube - ndarray = HyperspectralCube
            => requires the ndarray to be of broadcast-compatible shape.

        HyperspectralCube - HyperspectralCube = HyperspectralCube
            => requires the cubes to be of same shapes and same referentials.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot subtract from an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot subtract an empty HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot subtract HyperspectralCubes with "
                                "different referentials.")
            result = self.subtract(other)
            result.x = self.x
            result.y = self.y
            result.z = self.z
            return result
        else:
            # other types, numpy will raise when inappropriate
            data = self.data - other

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __rsub__(self, other):
        """
        Subtraction should work pretty much like numpy's ndarray subtraction.

        Number - HyperspectralCube = HyperspectralCube
            => subtract each voxel of the cube to the number, in each voxel

        ndarray - HyperspectralCube = ndarray
            => requires the ndarray to be of broadcast-compatible shape.
               The result is a ndarray. This can't be helped ; deal with it.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot subtract an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot subtract from an empty "
                                "HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot subtract HyperspectralCubes with "
                                "different referentials.")
            result = other.subtract(self)
            result.x = self.x
            result.y = self.y
            result.z = self.z
            return result
        else:
            # other types, numpy will raise when inappropriate
            data = other - self.data

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __mul__(self, other):
        """
        Addition should work pretty much like numpy's ndarray addition.

        HyperspectralCube * Number = HyperspectralCube
            => multiply each voxel of the cube by the number.

        HyperspectralCube * ndarray = HyperspectralCube
            => requires the ndarray to be of broadcast-compatible shape.

        ndarray * HyperspectralCube = ndarray
            => requires the ndarray to be of broadcast-compatible shape.
               The result is a ndarray. This can't be helped ; deal with it.

        HyperspectralCube * HyperspectralCube = HyperspectralCube
            => requires the cubes to be of same shapes and same referentials.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot multiply an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot multiply by an empty "
                                "HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot multiply HyperspectralCubes with "
                                "different referentials.")
            result = self.multiply(other)
            result.x = self.x
            result.y = self.y
            result.z = self.z
            return result
        else:
            # other types, numpy will raise when inappropriate
            data = self.data * other

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __rmul__(self, other):
        """
        Multiplication following Peano axioms is commutative on our sets.
        """
        return self.__mul__(other)

    def __div__(self, other):
        """
        Division should work pretty much like numpy's ndarray division.
        We don't check for division by 0, and let exceptions bubble from numpy.
        It's faster, tends to be IEEE 754 compliant, and is consistent with
        numpy.

        TODO: find out the potential impact of `from __future__ import division`
        on this, if any.

        HyperspectralCube / Number = HyperspectralCube
            => divide each voxel of the cube by the number.

        HyperspectralCube / ndarray = HyperspectralCube
            => requires the ndarray to be of broadcast-compatible shape.

        HyperspectralCube / HyperspectralCube = HyperspectralCube
            => requires the cubes to be of same shapes and same referentials.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot divide an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot divide by an empty HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot divide HyperspectralCubes with "
                                "different referentials.")
            result = self.divide(other)
            result.x = self.x
            result.y = self.y
            result.z = self.z
            return result
        else:
            # other types, numpy will raise when inappropriate
            data = self.data / other

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __rdiv__(self, other):
        """
        Division should work pretty much like numpy's ndarray division.
        We don't check for division by 0, and let exceptions bubble from numpy.
        It's faster, tends to be IEEE 754 compliant, and is consistent with
        numpy.

        TODO: find out the potential impact of `from __future__ import division`
        on this, if any.

        Number / HyperspectralCube = HyperspectralCube
            => divide the number by the voxel, in each voxel.

        ndarray / HyperspectralCube = ndarray
            => requires the ndarray to be of broadcast-compatible shape.
               The result is a ndarray. This can't be helped ; deal with it.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot divide by an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot divide an empty HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot divide HyperspectralCubes with "
                                "different referentials.")
            result = other.divide(self)
            result.x = self.x
            result.y = self.y
            result.z = self.z
            return result
        else:
            # other types, numpy will raise when inappropriate
            data = other / self.data

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __pow__(self, other):
        """
        Exponentiation should work pretty much like numpy's ndarray
        exponentiation.

        This is still a work in progress and full support it yet to be
        implemented mainstream (in astropy.nddata).

        HyperspectralCube ** Number = HyperspectralCube
            => power each voxel of the cube by the number.

        HyperspectralCube ** ndarray = HyperspectralCube
            => requires the ndarray to be of broadcast-compatible shape.

        HyperspectralCube ** HyperspectralCube = HyperspectralCube
            => NOPE. This is a WiP, and this will be implemented once NDData
               implements it, mainstream.
            => requires the cubes to be of same shapes and referentials.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Cannot exponentiate an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            if other.is_empty():
                raise TypeError("Cannot exponentiate by an empty "
                                "HyperspectralCube.")
            if not self.has_same_referential_as(other):
                raise TypeError("Cannot exponentiate HyperspectralCubes with "
                                "different referentials.")
            #result = self.exponentiate(other)  # <= not implemented yet
            # result.x = self.x
            # result.y = self.y
            # result.z = self.z
            # return result

            # Meanwhile, ...
            raise NotImplementedError("There is no mainstream support for "
                                      "exponentiation yet in "
                                      "astropy.nddata.NDData.")
        else:
            # other types, numpy will raise when inappropriate
            data = self.data ** other

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def __rpow__(self, other):
        """
        Exponentiation should work pretty much like numpy's ndarray
        exponentiation.

        This is still a work in progress and full support it yet to be
        implemented mainstream (in astropy.nddata).

        Number ** HyperspectralCube = HyperspectralCube
            => power number by the voxel, in each voxel.

        ndarray ** HyperspectralCube = ndarray
            => requires the ndarray to be of broadcast-compatible shape.
               The result is a ndarray. This can't be helped ; deal with it.

        Raises TypeErrors when cubes are empty or operands are not compatible.
        """
        if self.is_empty():
            raise TypeError("Can't exponentiate by an empty HyperspectralCube.")

        if isinstance(other, HyperspectralCube):
            raise NotImplementedError("Wait... WHAT ? This should use __pow__.")
        else:
            # other types, numpy will raise when inappropriate
            data = other ** self.data

        return HyperspectralCube(data=data, x=self.x, y=self.y, z=self.z)

    def has_same_referential_as(self, other):
        return \
            self.x.is_same_as(other.x) and \
            self.y.is_same_as(other.y) and \
            self.z.is_same_as(other.z)
