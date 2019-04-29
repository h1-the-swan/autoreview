# -*- coding: utf-8 -*-
#
import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

from .config import Config
from .util import load_random_state, prepare_directory, load_spark_dataframe, save_pandas_dataframe_to_pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Autoreview(object):

    """Toplevel Autoreview object"""

    def __init__(self, id_list, citations, papers, outdir, sample_size, random_seed=None, id_colname='UID', citing_colname=None, cited_colname='cited_UID', config=None):
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
        self.id_colname = id_colname
        if citing_colname is not None:
            self.citing_colname = citing_colname
        else:
            self.citing_colname = id_colname
        self.cited_colname = cited_colname

        if config is not None:
            assert isinstance(config, Config)
            self._config = config
        else:
            self._config = Config()
        self.spark = self._config.spark

    def follow_citations(self, sdf, sdf_citations):
        """follow in- and out-citations

        :sdf: a spark dataframe with one column `ID` that contains the ids to follow in- and out-citations
        :sdf_citations: spark dataframe with citation data. columns are `ID` and `cited_ID`
        :returns: spark dataframe with one column `ID` that contains deduplicated IDs for in- and out-citations

        """
        sdf_outcitations = sdf_citations.join(sdf, on='ID', how='inner')
        _sdf_renamed = sdf.withColumnRenamed('ID', 'cited_ID')
        sdf_incitations = sdf_citations.join(_sdf_renamed, on='cited_ID')

        sdf_combined = self.combine_ids([sdf_incitations, sdf_outcitations])
        return sdf_combined

    def combine_ids(self, sdfs):
        """Given a list of spark dataframes with columns ['UID', 'cited_UID']
        return a dataframe with one column 'UID' containing all of the UIDs in both columns of the input dataframes
        """
        sdf_combined = self.spark.createDataFrame([], schema='ID string')
        for sdf in sdfs:
            # add 'UID' column
            sdf_combined = sdf_combined.union(sdf.select(['ID']))
            
            # add 'cited_UID' column (need to rename)
            sdf_combined = sdf_combined.union(sdf.select(['cited_ID']).withColumnRenamed('cited_ID', 'ID'))
        return sdf_combined.drop_duplicates()

    def run(self):
        """Run and save output

        """
        try:
            prepare_directory(self.outdir)

            df_id_list = pd.DataFrame(self.id_list, columns=['ID'])
            seed_papers, target_papers = train_test_split(df_id_list, train_size=self.sample_size, random_state=self.random_state)

            sdf_papers = load_spark_dataframe(self.papers, self.spark) 
            sdf_papers = sdf_papers.withColumnRenamed(self.id_colname, 'ID')

            sdf_seed = self.spark.createDataFrame(seed_papers[['ID']])
            sdf_target = self.spark.createDataFrame(target_papers[['ID']])
            # self.sdf_seed_count = sdf_seed.count()
            # self.sdf_target_count = sdf_target.count()

            outfname = os.path.join(self.outdir, 'seed_papers.pickle')
            logger.debug('saving seed papers to {}'.format(outfname))
            start = timer()
            df_seed = sdf_seed.join(sdf_papers, on='ID', how='inner').toPandas()
            save_pandas_dataframe_to_pickle(df_seed, outfname)
            logger.debug("done saving seed papers. took {}".format(format_timespan(timer()-start)))

            outfname = os.path.join(self.outdir, 'target_papers.pickle')
            logger.debug('saving target papers to {}'.format(outfname))
            start = timer()
            df_target = sdf_target.join(sdf_papers, on='ID', how='inner').toPandas()
            save_pandas_dataframe_to_pickle(df_target, outfname)
            logger.debug("done saving target papers. took {}".format(format_timespan(timer()-start)))

            sdf_citations = load_spark_dataframe(self.citations, self.spark)
            sdf_citations = sdf_citations.withColumnRenamed(self.citing_colname, 'ID')
            sdf_citations = sdf_citations.withColumnRenamed(self.cited_colname, 'cited_ID')

            # collect IDs for in- and out-citations
            sdf_combined = self.follow_citations(sdf_seed, sdf_citations)
            # do it all again to get second degree
            sdf_combined = self.follow_citations(sdf_combined, sdf_citations)

            df_combined = sdf_combined.join(sdf_papers, on='ID', how='inner').toPandas()
            save_pandas_dataframe_to_pickle(df_combined, os.path.join(self.outdir, 'TESTTESTPAPERS.pickle'))
        
        finally:
            self._config.teardown()
