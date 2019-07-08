# -*- coding: utf-8 -*-

from .context import autoreview

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_test_works(self):
        assert True


if __name__ == '__main__':
    unittest.main()
