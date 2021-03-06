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
            line = line.strip()
            if line:
                id_list.append(line)
    return id_list

class SparkMAGTestSuite(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.config = autoreview.Config()
        from dotenv import load_dotenv
        load_dotenv(os.path.join(self.config.PROJECT_DIR, 'tests', 'test.env'))
        self.config.PATH_TO_PAPER_DATA = os.environ['MAG_PATH_TO_PAPER_DATA']
        self.config.PATH_TO_CITATION_DATA = os.environ['MAG_PATH_TO_CITATION_DATA']
        id_list_fname = os.path.join(self.config.PROJECT_DIR, 'tests',  'test_id_list_mag.txt')
        self.id_list = get_id_list(id_list_fname)
        self.outdir = os.path.join(self.config.PROJECT_DIR, 'tests', 'test_outdir_mag')

    def test_id_list(self):
        self.assertEqual(len(self.id_list), 7)
        for item in self.id_list:
            self.assertIsInstance(item, str)

    def test_autoreview_run(self):
        autorev = autoreview.Autoreview(id_list=self.id_list,
                citations=self.config.PATH_TO_CITATION_DATA,
                papers=self.config.PATH_TO_PAPER_DATA,
                outdir=self.outdir,
                sample_size=3,
                random_seed=999,
                id_colname='Paper_ID',
                cited_colname='Paper_reference_ID',
                config=self.config)
        self.assertIs(os.path.exists(self.outdir), False)
        autorev.run()
        self.assertIs(os.path.exists(self.outdir), True)

    def tearDown(self):
        # if os.path.exists(self.outdir):
        #     shutil.rmtree(self.outdir)
        pass


if __name__ == '__main__':
    unittest.main()
