# -*- coding: utf-8 -*-

from .context import autoreview

import os, shutil
import unittest

def get_id_list(fname):
    """Get a list of IDs from a newline separated file
    """
    id_list = []
    with open(fname, 'r') as f:
        for line in f:
            id_list.append(line.strip())
    return id_list

class SparkTestSuite(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.config = autoreview.Config()
        id_list_fname = os.path.join(self.config.PROJECT_DIR, 'tests',  'test_id_list.txt')
        self.id_list = get_id_list(id_list_fname)
        self.outdir = os.path.join(self.config.PROJECT_DIR, 'tests', 'test_outdir')

    def test_autoreview_run(self):
        autorev = autoreview.Autoreview(id_list=self.id_list,
                citations=self.config.PATH_TO_CITATION_DATA,
                papers=self.config.PATH_TO_PAPER_DATA,
                outdir=self.outdir,
                sample_size=3,
                random_seed=999,
                config=self.config)
        autorev.run()

    def tearDown(self):
        shutil.rmtree(self.outdir)


if __name__ == '__main__':
    unittest.main()
