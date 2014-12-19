# coding=utf-8

## GENERAL PACKAGES
import os
from os.path import abspath, join, dirname
import logging
import numpy
import unittest
from ddt import ddt, data

import astropy.units as u

from hyperspectral import HyperspectralCube, Axis
from numpy_ndarray_assertions import NumpyNdArrayAssertions

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('HyperspectralCubeTest')


# We're using Data-Driven Testing extensively here, in all tests annotated with
# @data(*filepaths). Simply add the FITS file that this library fails to parse
# into the test/data directory, and run this test suite again.
# You are encouraged to add more test cases. Tests, tests, TESTS !

# Mandatory reading : http://docs.python-guide.org/en/latest/writing/tests/
# Have a look at http://ddt.readthedocs.org/en/latest for the @data annotation.


def filepaths_provider():
    """
    Collects filepaths of test FITS files located in `test/data/`.
    Some tests here will be run against each of these files.
    Test cases annotated by @data will be run multiple times, once per filepath.
    """
    root_folder = dirname(abspath(__file__))
    fits_folder = join(root_folder, 'data')
    _filepaths = []
    for filename in os.listdir(fits_folder):
        filepath = os.path.join(fits_folder, filename)
        if os.path.isfile(filepath) and filepath.endswith('.fits'):
            _filepaths.append(filepath)
    return _filepaths

filepaths = filepaths_provider()


@ddt
class HyperspectralCubeTest(unittest.TestCase, NumpyNdArrayAssertions):

    longMessage = True

    root_folder = dirname(abspath(__file__))
    fits_folder = join(root_folder, 'data')

    muse_01_filename = join(fits_folder, 'muse_01.fits')

    ## SANITY CHECKS ###########################################################

    def test_init_empty_cube(self):
        cube = HyperspectralCube()
        self.assertTrue(cube.is_empty())

    def test_init_with_unfit_filename(self):
        with self.assertRaises(IOError):
            cube = HyperspectralCube.from_fits('this_is_not_a_fits_file')

    ## PUBLIC PROPERTIES #######################################################

    def test_shape_property(self):
        data_a = numpy.array([
            [[0, 1], [2, 3], [-1, +0]],
            [[1, 0], [3, 2], [+3, +2]],
            [[1, 1], [2, 2], [-3, +0]],
        ], dtype=float)
        cube_a = HyperspectralCube(data=data_a)

        self.assertArrayEqual(
            (3, 3, 2), cube_a.shape,
            "cube.shape should return the shape"
        )

    ## SLICING AND INDEXING ####################################################

    def test_slicing_and_indexing(self):
        # Dummy data
        data_a = numpy.array([
            [[0, 1], [2, 3], [-1, 0]],
            [[1, 0], [3, 2], [+3, 2]],
            [[1, 1], [2, 2], [-3, 0]],
        ], dtype=float)

        x = Axis('x', 0., 1.)
        y = Axis('y', 1., 2.)
        z = Axis('z', 2., 3.)

        cube_a = HyperspectralCube(data=data_a, x=x, y=y, z=z)

        # SLICING USING [:,:,:]
        cube_t = cube_a[0:1, :, :-1]
        data_t = numpy.array([[[0.], [2.], [-1.]]], dtype=float)
        self.assertIsInstance(cube_t, HyperspectralCube,
                              "Truncation result is a HyperspectralCube")
        self.assertArrayEqual(data_t, cube_t.data,
                              "Supports extraction of sub-cube using [:,:,:]")
        # SLICING UPDATES METADATA
        self.assertEqual(cube_t.z.start, z.start, "Z axis start is unchanged")
        self.assertEqual(cube_t.y.start, y.start, "Y axis start is unchanged")
        self.assertEqual(cube_t.x.start, x.start, "X axis start is unchanged")

        # SLICING [min:max] <=> [min:max,:,:]
        cube_t = cube_a[1:2]
        data_t = numpy.array([[[1, 0], [3, 2], [+3, 2]]], dtype=float)
        self.assertArrayEqual(data_t, cube_t.data,
                              "Unspecified axes are defaulted to `:` in slicing")
        # SLICING UPDATES METADATA
        self.assertEqual(cube_t.z.start, z.start + z.step,
                         "Z axis start has shifted")

        # MUTATING ONE VALUE USING [λ,y,x] INDICES
        data_t = numpy.array([
            [[0, 1], [2, 3], [-1, 0]],
            [[1, 0], [3, 9], [+3, 2]],
            [[1, 1], [2, 2], [-3, 0]],
        ], dtype=float)
        cube_t = cube_a.copy()
        cube_t[1, 1, 1] = 9
        self.assertArrayEqual(data_t, cube_t.data,
                              "Supports mutation cube[λ,y,x] = n")

        # MUTATING MULTIPLE VALUES USING [:,y,x] INDICES
        data_t = numpy.array([
            [[0, 1], [2, 9], [-1, 0]],
            [[1, 0], [3, 9], [+3, 2]],
            [[1, 1], [2, 9], [-3, 0]],
        ], dtype=float)
        cube_t = cube_a.copy()
        cube_t[:, 1, 1] = 9
        self.assertArrayEqual(data_t, cube_t.data,
                              "Supports mutation cube[:,y,x] = n")

    ## PUBLIC METHODS ##########################################################

    def test_getting_steps(self):
        # This test is hardwired to muse_01.fits
        cube = HyperspectralCube.from_fits(self.muse_01_filename)

        # Collect the data
        steps = cube.get_steps()
        # print "STEPS: %s" % steps
        # <Quantity 1.25 Angstrom>
        # <Quantity 5.55555555556e-05 deg>
        # <Quantity -5.55555555556e-05 deg>

        # Test the success/failure of the API
        self.assertEqual(steps[0], cube.get_step(0))
        self.assertEqual(steps[1], cube.get_step(1))
        self.assertEqual(steps[2], cube.get_step(2))

        # Test unit conversion -- this test is hardwired to muse_01.fits
        self.assertEqual(cube.get_step(0).value, 1.25)
        self.assertAlmostEqual(
            cube.get_step(0).to(u.Unit('micron')).value,
            1.25e-4
        )

    def test_wavelength_of(self):
        # This test is hardwired to muse_01.fits
        cube = HyperspectralCube.from_fits(self.muse_01_filename)

        # With a singe pixel position
        self.assertEqual(u.Quantity(6734.7802734375, 'Angstrom'),
                         cube.wavelength_of(3))

        # With a list of pixel positions
        self.assertArrayEqual([6734.7802734375, 6737.2802734375],
                              cube.wavelength_of([3, 5]).value)
        self.assertEqual(u.Unit('Angstrom'), cube.wavelength_of([3, 5]).unit)

        # Ideally, we'd run the following test, but...
        # "The truth value of an array with more than one element is ambiguous."
        # and assertArrayEqual makes a implicit ndarray conversion that logs
        # warnings all over the place.
        # self.assertEqual(
        #     u.Quantity([6734.7802734375, 6737.2802734375], 'Angstrom'),
        #     cube.wavelength_of([3, 5])
        # )

    def test_pixel_of(self):
        # This test is hardwired to muse_01.fits
        cube = HyperspectralCube.from_fits(self.muse_01_filename)

        # With a single wavelength expressed as a simple float
        self.assertEqual(3, cube.pixel_of(6734.7802734375))

        # With a list of wavelengths
        self.assertArrayEqual(
            [3, 5],
            cube.pixel_of([6734.7802734375, 6737.2802734375]),
            "pixel_of() supports a list as input"
        )

        # With a ndarray of wavelengths
        self.assertArrayEqual(
            numpy.array([3, 5]),
            cube.pixel_of(numpy.array([6734.7802734375, 6737.2802734375])),
            "pixel_of() supports a numpy array as input"
        )

    # def test_write_to_file(self):
    #     cube = HyperspectralCube.from_fits(self.fits_test_filename)
    #     fits_out_filename = os.path.join(self.fits_folder, 'tmp_output.fits')
    #
    #     self.assertFalse(os.path.isfile(fits_out_filename),
    #                      "Sanity check : Output FITS file should not exist ; "
    #                      "Please remove it by hand and then re-run this test : %s" % fits_out_filename)
    #     cube.write_to(fits_out_filename)
    #     self.assertTrue(os.path.isfile(fits_out_filename),
    #                     "Output FITS file should be created")
    #
    #     with self.assertRaises(IOError):
    #         cube.write_to(fits_out_filename)  # clobber option should be false by default
    #
    #     cube.write_to(fits_out_filename, clobber=True)  # overwrites without raising
    #     os.remove(fits_out_filename)  # cleanup

    ## DATA-DRIVEN TESTS #######################################################

    @data(*filepaths)
    def test_from_fits(self, filepath):
        cube = HyperspectralCube.from_fits(filepath)  # should not raise
        print cube

    @data(*filepaths)
    def test_wavelength_conversion_roundtrip(self, filepath):
        cube = HyperspectralCube.from_fits(filepath)

        # For each spectral pixel index
        for pixel in range(0, cube.shape[0]):
            self.assertEqual(
                round(cube.pixel_of(cube.wavelength_of(pixel))),
                pixel,
                "Wavelength conversion from pixel index #%d to λ "
                "and back from λ to pixel index #%d." % (pixel, pixel)
            )

    ## ARITHMETICS #############################################################

    def test_arithmetic_core(self):
        inf = numpy.inf
        nan = numpy.nan

        # TEST DATA
        data_a = numpy.array([
            [[0, 1], [2, 3], [-1, +0]],
            [[1, 0], [3, 2], [+3, +2]],
            [[1, 1], [2, 2], [-3, +0]],
        ], dtype=float)
        data_b = numpy.array([
            [[1, 1], [0, 0], [+1, -1]],
            [[0, 0], [3, 1], [+3, -2]],
            [[2, 2], [1, 2], [-1, +0]],
        ], dtype=float)
        data_i = numpy.array([
            [[2, 0], [1, 1], [+1, -1]],
        ], dtype=float)

        # BRAIN COMPUTED EXPECTATIONS
        data_a_plus_1 = numpy.array([
            [[1, 2], [3, 4], [+0, +1]],
            [[2, 1], [4, 3], [+4, +3]],
            [[2, 2], [3, 3], [-2, +1]],
        ], dtype=float)
        data_a_plus_b = numpy.array([
            [[1, 2], [2, 3], [+0, -1]],
            [[1, 0], [6, 3], [+6, -0]],
            [[3, 3], [3, 4], [-4, +0]],
        ], dtype=float)
        data_a_plus_i = numpy.array([
            [[2, 1], [3, 4], [+0, -1]],
            [[3, 0], [4, 3], [+4, +1]],
            [[3, 1], [3, 3], [-2, -1]],
        ], dtype=float)
        data_a_minus_1 = numpy.array([
            [[-1, +0], [1, 2], [-2, -1]],
            [[+0, -1], [2, 1], [+2, +1]],
            [[+0, +0], [1, 1], [-4, -1]],
        ], dtype=float)
        data_1_minus_a = numpy.array([
            [[+1, +0], [-1, -2], [+2, +1]],
            [[+0, +1], [-2, -1], [-2, -1]],
            [[+0, +0], [-1, -1], [+4, +1]],
        ], dtype=float)
        data_a_times_2 = numpy.array([
            [[0, 2], [4, 6], [-2, +0]],
            [[2, 0], [6, 4], [+6, +4]],
            [[2, 2], [4, 4], [-6, +0]],
        ], dtype=float)
        data_a_times_b = numpy.array([
            [[0, 1], [0, 0], [-1, +0]],
            [[0, 0], [9, 2], [+9, -4]],
            [[2, 2], [2, 4], [+3, +0]],
        ], dtype=float)
        data_a_div_2 = numpy.array([
            [[.0, .5], [1.0,  3./2], [-.5,   0]],
            [[.5, .0], [3./2,  1.0], [3./2,  1]],
            [[.5, .5], [1.0,   1.0], [-3./2, 0]],
        ], dtype=float)
        data_a_div_b = numpy.array([
            [[0,     1], [inf, inf], [-1,  -0.]],
            [[inf, nan], [1,     2], [+1,  -1.]],
            [[.5,   .5], [2,     1], [+3,  nan]],
        ], dtype=float)
        data_a_pow_2 = numpy.array([
            [[0, 1], [4, 9], [+1, +0]],
            [[1, 0], [9, 4], [+9, +4]],
            [[1, 1], [4, 4], [+9, +0]],
        ], dtype=float)
        data_2_pow_a = numpy.array([
            [[1, 2], [4, 8], [0.5,  1]],
            [[2, 1], [8, 4], [8,    4]],
            [[2, 2], [4, 4], [1./8, 1]],
        ], dtype=float)
        data_b_pow_a = numpy.array([
            [[1, 1], [0,  0], [+1, +1]],
            [[0, 1], [27, 1], [27, +4]],
            [[2, 2], [1,  4], [-1, +1]],
        ], dtype=float)

        self.assertArrayEqual(data_a_div_b, data_a_div_b,
                              "Sanity check with inf and nan")

        cube_a = HyperspectralCube(data=data_a)
        cube_b = HyperspectralCube(data=data_b)
        cube_e = HyperspectralCube()             # empty

        # CUBE + NUMBER
        cube_a_plus_1 = cube_a + 1.
        self.assertIsInstance(cube_a_plus_1, HyperspectralCube,
                              "Addition result is a HyperspectralCube")
        self.assertArrayEqual(data_a_plus_1, cube_a_plus_1.data,
                              "Supports addition of cube and number using +")

        # # NUMBER + CUBE
        cube_a_plus_1 = 1. + cube_a
        self.assertIsInstance(cube_a_plus_1, HyperspectralCube,
                              "Addition result is a HyperspectralCube")
        self.assertArrayEqual(data_a_plus_1, cube_a_plus_1.data,
                              "Supports addition of number and cube using +")

        # CUBE A + CUBE B
        cube_a_plus_b = cube_a + cube_b
        self.assertIsInstance(cube_a_plus_b, HyperspectralCube,
                              "Addition result is a HyperspectralCube")
        self.assertArrayEqual(data_a_plus_b, cube_a_plus_b.data,
                              "Supports addition of two cubes using +")

        # CUBE A + NDARRAY
        cube_a_plus_b = cube_a + data_b
        self.assertIsInstance(cube_a_plus_b, HyperspectralCube,
                              "Addition result is a HyperspectralCube")
        self.assertArrayEqual(data_a_plus_b, cube_a_plus_b.data,
                              "Supports addition of cube and ndarray using +")

        # CUBE + IMAGE
        cube_a_plus_i = cube_a + data_i
        self.assertIsInstance(cube_a_plus_i, HyperspectralCube,
                              "Addition result is a HyperspectralCube")
        self.assertArrayEqual(data_a_plus_i, cube_a_plus_i.data,
                              "Supports addition of cube and image using +")

        # IMAGE + CUBE
        # This is tricky : it yields a numpy.ndarray because numpy is flexible
        # in what it accepts as the right hand operator, and it accepts our
        # HyperspectralCube because it provides the `shape` property, so our
        # __radd__ method is never even called.
        cube_a_plus_i = data_i + cube_a
        self.assertIsInstance(cube_a_plus_i, numpy.ndarray,
                              "Addition result is a numpy.ndarray")
        self.assertArrayEqual(data_a_plus_i, cube_a_plus_i,
                              "Supports addition of image and cube using +")

        # ADDITION EXPECTED ERRORS
        with self.assertRaises(TypeError):
            cube_a + cube_e  # right-hand empty cube
        with self.assertRaises(TypeError):
            cube_e + cube_b  # left-hand empty cube
        with self.assertRaises(TypeError):
            cube_b + 'rock'  # left-hand "garbage"
        with self.assertRaises(TypeError):
            '666.' + cube_a  # right-hand "garbage"

        # CUBE - NUMBER
        cube_a_minus_1 = cube_a - 1
        self.assertIsInstance(cube_a_minus_1, HyperspectralCube,
                              "Subtraction result is a HyperspectralCube")
        self.assertArrayEqual(
            data_a_minus_1, cube_a_minus_1.data,
            "Supports subtraction of number from cube using -"
        )

        # NUMBER - CUBE
        cube_1_minus_a = 1 - cube_a
        self.assertIsInstance(cube_1_minus_a, HyperspectralCube,
                              "Subtraction result is a HyperspectralCube")
        self.assertArrayEqual(
            data_1_minus_a, cube_1_minus_a.data,
            "Supports subtraction of cube from number using -"
        )

        # CUBE * NUMBER
        cube_a_times_2 = cube_a * 2
        self.assertIsInstance(cube_a_times_2, HyperspectralCube,
                              "Multiplication result is a HyperspectralCube")
        self.assertArrayEqual(
            data_a_times_2, cube_a_times_2.data,
            "Supports multiplication of cube and number using *"
        )

        # NUMBER * CUBE
        cube_a_times_2 = 2 * cube_a
        self.assertIsInstance(cube_a_times_2, HyperspectralCube,
                              "Multiplication result is a HyperspectralCube")
        self.assertArrayEqual(
            data_a_times_2, cube_a_times_2.data,
            "Supports multiplication of number and cube using *"
        )

        # CUBE A * CUBE B
        cube_a_times_b = cube_a * cube_b
        self.assertIsInstance(cube_a_times_b, HyperspectralCube,
                              "Multiplication result is a HyperspectralCube")
        self.assertArrayEqual(data_a_times_b, cube_a_times_b.data,
                              "Supports multiplication of two cubes using *")

        # CUBE / NUMBER
        cube_a_div_2 = cube_a / 2
        self.assertIsInstance(cube_a_div_2, HyperspectralCube,
                              "Division result is a HyperspectralCube")
        self.assertArrayEqual(data_a_div_2, cube_a_div_2.data,
                              "Supports division of cube and number using /")

        # CUBE / CUBE
        cube_a_div_b = cube_a / cube_b
        self.assertIsInstance(cube_a_div_b, HyperspectralCube,
                              "Division result is a HyperspectralCube")
        self.assertArrayEqual(data_a_div_b, cube_a_div_b.data,
                              "Supports division of two cubes using /")

        # CUBE ** NUMBER
        cube_a_pow_2 = cube_a ** 2
        self.assertIsInstance(cube_a_pow_2, HyperspectralCube,
                              "Exponentiation result is a HyperspectralCube")
        self.assertArrayEqual(
            data_a_pow_2, cube_a_pow_2.data,
            "Supports exponentiation of cube by number using **"
        )

        # NUMBER ** CUBE
        cube_2_pow_a = 2 ** cube_a
        self.assertIsInstance(cube_2_pow_a, HyperspectralCube,
                              "Exponentiation result is a HyperspectralCube")
        self.assertArrayEqual(
            data_2_pow_a, cube_2_pow_a.data,
            "Supports exponentiation of number by cube using **"
        )

        # CUBE A ** CUBE B -- wait for mainstream implementation first
        # cube_b_pow_a = cube_b ** cube_a
        # self.assertIsInstance(cube_b_pow_a, HyperspectralCube,
        #                       "Exponentiation result is a HyperspectralCube")
        # self.assertArrayEqual(
        #     data_b_pow_a, cube_b_pow_a.data,
        #     "Supports exponentiation of cube by cube using **"
        # )


if __name__ == "__main__":
    unittest.main()