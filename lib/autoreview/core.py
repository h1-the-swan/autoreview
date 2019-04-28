# -*- coding: utf-8 -*-
#
from .config import Config
from .util import load_random_state, prepare_directory, load_spark_dataframe

from sklearn.model_selection import train_test_split

class Autoreview(object):

    """Toplevel Autoreview object"""

    def __init__(self, id_list, citations, papers, outdir, sample_size, random_seed=None, config=None):
        """
        :id_list: list of strings: IDs for the seed set
        :citations: path to citations data
        :papers: path to papers data
        :outdir: output directory. will raise RuntimeError if the directory already exists
        :sample_size: integer: size of the seed set to split off from the initial. the rest will be used as target papers
        :random_seed: integer

        """
        self.id_list = id_list
        self.citations = citations
        self.papers = papers
        self.outdir = outdir
        self.sample_size = sample_size
        self.random_state = load_random_state(random_seed)

        if config is not None:
            assert isinstance(config, Config)
            self._config = config
        else:
            self._config = Config()
        self.spark = self._config.spark

    def run(self):
        """Run and save output

        """
        try:
            prepare_directory(self.outdir)

            seed_papers, target_papers = train_test_split(self.id_list, train_size=self.sample_size, random_state=self.random_state)

            sdf_papers = load_spark_dataframe(self.papers, self.spark) 
        
        finally:
            self._config.teardown()
