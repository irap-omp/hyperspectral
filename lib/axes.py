from astropy.units import Unit, Quantity


class Axis:
    """
    Describes an axis in a multidimensional data space.
    - name : a descriptive name for the axis.
    - start : the value on the axis of the zeroth pixel.
    - step : the step on this axis from pixel N to pixel N+1.
    - unit : the unit of the axis, an instance of `astropy.units.Unit`.
             if not specified, will be the dimensionless unit
    """

    def __init__(self, name, start, step, unit=None):
        if unit is None:
            unit = Unit('')  # note: null unit `Unit()` from doc does not work
        self.name = name
        self.start = Quantity(start, unit)
        self.step = Quantity(step, unit)
        self.unit = unit

    def __str__(self):
        s = "Axis {a.name}, start {a.start.value}, step {a.step.value}"
        if self.unit is not '':  # non-trivial implicit string casting with `is`
            s += ", in {a.unit}"
        s += "."
        return s.format(a=self)

    def copy(self, out=None):
        """
        Copies this object in out (if specified) and returns the copy.
        """
        out = Axis(self.name, self.start, self.step, self.unit)
        return out

    def is_same_as(self, other):
        """
        Compares this axis with an *other* axis, return True if they're similar.
        Maybe this ought to be __eq__ ?
        """
        return \
            self.start == other.start and \
            self.step == other.step and \
            self.unit == other.unit


class UndefinedAxis(Axis):
    def __init__(self, name):
        Axis.__init__(self, name, 0., 1.)

    def __str__(self):
        return 'Axis {a.name} is undefined.'.format(a=self)