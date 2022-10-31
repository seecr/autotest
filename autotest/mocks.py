

""" this should be a hook, so you can get a mock by:

    m = test.mock_object(...)

    Doing so keeps test the only API access point.
"""

# helpers
def mock_object(*functions, **more):
    """ Creates an object from a bunch of functions.
        Useful for testing methods from inside the class definition. """
    self = mock.Mock()
    self.configure_mock(**{f.__name__: types.MethodType(f, self) for f in functions}, **more)
    return self



