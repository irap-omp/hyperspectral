import numpy


class NumpyNdArrayAssertions():
    """
    This class is a mixin following the UnitTest naming conventions.
    It is meant to be used along with unittest.TestCase like so :
    class MyTestCase(unittest.TestCase, NumpyNdArrayAssertions):
        ...
    It needs python >= 2.6 and numpy >= 1.6
    """

    def assertArrayEqual(self, expected_array, actual_array, msg=None):
        """
        Fail if actual array is not the same as expected array,
        both element-wise and shape-wise, and don't choke on NaNs.
        """
        stdmsg = "%s\nis not expected\n%s" % (str(actual_array), str(expected_array))
        numpy.testing.assert_array_equal(actual_array, expected_array,
                                         err_msg=self._formatMessage(msg, stdmsg),
                                         verbose=self.longMessage)

    def assertArrayEquals(self, expected_array, actual_array, msg=None):
        """
        Alias for assertArrayEqual
        """
        self.assertArrayEqual(expected_array, actual_array, msg)

    def assertArrayEqualish(self, expected_array, actual_array, expected_similarity=99., msg=None):
        """
        Fails if average similarity is less than expected_similarity (in %)
        """
        a = numpy.nan_to_num(actual_array)
        e = numpy.nan_to_num(expected_array)

        actual_precision = 100. * (1 - numpy.nan_to_num(numpy.abs(e - a) / e))
        actual_precision = numpy.sum(actual_precision) / actual_precision.size

        standardMsg = "%f %% is less than %f %%" % (actual_precision, expected_similarity)

        self.assertTrue(actual_precision >= expected_similarity, self._formatMessage(msg, standardMsg))

        return actual_precision