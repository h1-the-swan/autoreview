# -*- coding: utf-8 -*-

import os
import shutil
from datetime import datetime
from pathlib import Path
from .context import autoreview

import pandas as pd
import numpy as np

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.config = autoreview.Config()
        # self.spark = self.config.spark
        self.paper_ids = Path('sample_data/sample_IDs_MAG.txt').read_text().split('\n')
        self.citations_fpath = 'sample_data/MAG_citations_sample/'
        self.papers_fpath = 'sample_data/MAG_papers_sample/'

        self.outdir = Path('tests/test_outdir')
        self.outdir.mkdir()

    def load_autorev(self):
        autorev = autoreview.Autoreview(id_list=self.paper_ids,
                                citations=self.citations_fpath,
                                papers=self.papers_fpath,
                                outdir=str(self.outdir),
                                sample_size=15,
                                random_seed=1,
                                id_colname='Paper_ID',
                                cited_colname='Paper_Reference_ID',
                                use_spark=False)
        return autorev

    def test_test_works(self):
        assert True

    def test_paper_ids_exists(self):
        assert len(self.paper_ids) > 0

    def test_initialize_autoreview(self):
        autorev = self.load_autorev()
        assert autorev.outdir == str(self.outdir)

    def test_get_papers_2_degrees_out(self):
        autorev = self.load_autorev()
        seed_papers, target_papers, candidate_papers = autorev.get_papers_2_degrees_out(use_spark=False)
        for df in [seed_papers, target_papers, candidate_papers]:
            assert not df.empty
        for pickle_file in ['seed_papers.pickle', 'target_papers.pickle', 'test_papers.pickle']:
            assert self.outdir.joinpath(pickle_file).exists()

    def test_train_models(self):
        autorev = self.load_autorev()
        seed_papers, target_papers, candidate_papers = autorev.get_papers_2_degrees_out(use_spark=False)
        autorev.train_models(seed_papers, target_papers, candidate_papers)
        assert len(list(self.outdir.rglob('best_model.pickle'))) > 0

    def test_year_lowpass_filter(self):
        year_lowpass_filter = autoreview.util.year_lowpass_filter
        papers = pd.read_parquet(self.papers_fpath)
        num_before = len(papers)
        papers = year_lowpass_filter(papers, year=1960, year_colname='year')
        num_after = len(papers)
        assert num_after == 3

    def test_prepare_data(self):
        autorev = self.load_autorev()
        seed_papers, target_papers, candidate_papers = autorev.get_papers_2_degrees_out(use_spark=False)
        candidate_papers, seed_papers, target_papers = autoreview.util.prepare_data_for_model(self.outdir, id_colname='ID', year=datetime.now().year, title_colname='title')
        assert not candidate_papers.empty
        assert not seed_papers.empty
        assert not target_papers.empty


    # def test_spark_working(self):
    #     assert self.spark is not None
    #     spark = self.spark
    #     assert spark is not None
    #
    # def test_load_papers(self):
    #     sdf_papers = self.spark.read.parquet(os.path.join('sample_data', 'MAG_papers_sample'))
    #     assert 'Paper_ID' in sdf_papers.columns
    #     assert 'cl' in sdf_papers.columns
    #     assert 'EF' in sdf_papers.columns
    #
    # def test_load_citations(self):
    #     sdf_citations = self.spark.read.parquet(os.path.join('sample_data', 'MAG_citations_sample'))
    #     assert 'Paper_ID' in sdf_citations.columns
    #     assert 'Paper_Reference_ID' in sdf_citations.columns
    #
    def tearDown(self):
        self.config.teardown()
        shutil.rmtree(self.outdir)



if __name__ == '__main__':
    unittest.main()
