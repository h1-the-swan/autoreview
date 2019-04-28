# -*- coding: utf-8 -*-

from .context import autoreview

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.config = autoreview.Config()
        self.spark = self.config.spark

    def test_spark_init(self):
        self.assertIsNotNone(self.spark)


if __name__ == '__main__':
    unittest.main()
