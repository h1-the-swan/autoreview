# -*- coding: utf-8 -*-

import os
from .context import autoreview

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.config = autoreview.Config()
        self.spark = self.config.spark

    def test_test_works(self):
        assert True

    def test_spark_working(self):
        assert self.spark is not None
        spark = self.spark
        assert spark is not None

    def test_load_papers(self):
        sdf_papers = self.spark.read.parquet(os.path.join('sample_data', 'MAG_papers_sample'))
        assert 'Paper_ID' in sdf_papers.columns
        assert 'cl' in sdf_papers.columns
        assert 'EF' in sdf_papers.columns

    def test_load_citations(self):
        sdf_citations = self.spark.read.parquet(os.path.join('sample_data', 'MAG_citations_sample'))
        assert 'Paper_ID' in sdf_citations.columns
        assert 'Paper_Reference_ID' in sdf_citations.columns

    def tearDown(self):
        self.config.teardown()



if __name__ == '__main__':
    unittest.main()
