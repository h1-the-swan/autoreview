# -*- coding: utf-8 -*-

DESCRIPTION = """Given a paper ID, get all of the outgoing references, then collect seed, target, and candidate sets for different random seeds based on those references."""

import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

from slugify import slugify
from autoreview import Autoreview
from autoreview.util import prepare_directory, load_spark_dataframe

from autoreview.config import Config

class PaperCollector(object):

    def __init__(self, config, basedir, paper_id, citations, papers, sample_size, random_seed, id_colname, cited_colname):
        """

        :config: Config object

        """
        self._config = config
        self.spark = config.spark
        self.basedir = basedir
        self.paper_id = paper_id
        self.citations = citations
        self.papers = papers
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.id_colname = id_colname
        self.cited_colname = cited_colname

        self.basedir = Path(self.basedir).resolve()
        if self.basedir.is_dir():
            logger.debug("Output basedir {} exists. Using this directory.".format(self.basedir))
        else:
            logger.debug("Output basedir {} does not exist. Creating it...".format(self.basedir))
            self.basedir.mkdir(parents=True)
        paper_id_slug = slugify(paper_id, lowercase=False)
        self.outdir = self.basedir.joinpath(paper_id_slug)
        if self.outdir.is_dir():
            logger.debug("Output directory {} exists. Using this directory.".format(self.outdir))
        else:
            logger.debug("Output directory {} does not exist. Creating it...".format(self.outdir))
            self.outdir.mkdir()

    def get_reference_ids(self, paper_id):
        """Use spark to get references

        :returns: list of paper IDs

        """
        sdf_citations = load_spark_dataframe(self.citations, self.spark)
        logger.debug("collecting references of paper ID: {}".format(paper_id))
        reference_ids = sdf_citations[sdf_citations[self.id_colname]==paper_id]
        logger.debug("count is {}".format(reference_ids.count()))
        reference_ids = reference_ids.toPandas()
        reference_ids = reference_ids[self.cited_colname].tolist()
        return reference_ids

    def main(self, args):
        reference_ids = self.get_reference_ids(self.paper_id)
        outfile = self.outdir.joinpath('reference_ids.csv')
        logger.debug("writing {} reference ids to {}".format(len(reference_ids), outfile))
        outfile.write_text("\n".join(reference_ids))
        # Autoreview object will make a new Config, so tear down the existing one
        # TODO: allow Autoreview to be passed an existing Config
        self._config.teardown()
        a = Autoreview(id_list=reference_ids,
                        citations=self.citations,
                        papers=self.papers,
                        outdir=self.outdir,
                        sample_size=self.sample_size,
                        random_seed=self.random_seed,
                        id_colname=self.id_colname,
                        cited_colname=self.cited_colname)
        a.get_papers_2_degrees_out()


def main(args):
    config = Config()
    pc = PaperCollector(config,
                basedir = args.basedir,
                paper_id = args.paper_id,
                citations=args.citations,
                papers=args.papers,
                sample_size=args.sample_size,
                random_seed=args.random_seed,
                id_colname=args.id_colname,
                cited_colname=args.cited_colname)
    pc.main(args)


if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("basedir", help="output base directory (will be created if it doesn't exist)")
    parser.add_argument("paper_id", help="paper ID to start with")
    parser.add_argument("--citations", help="citations data (to be read by spark)")
    parser.add_argument("--papers", help="papers/cluster data (to be read by spark)")
    parser.add_argument("--sample-size", type=int, default=200, help="number of articles to sample from the set to use to train the model (integer, default: 200)")
    parser.add_argument("--random-seed", type=int, default=999, help="random seed (integer, default: 999)")
    parser.add_argument("--id-colname", default='UID', help="column name for paper id (default: \"UID\")")
    parser.add_argument("--cited-colname", default='cited_UID', help="column name for cited paper id (default: \"cited_UID\")")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        logger.setLevel(logging.INFO)
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
