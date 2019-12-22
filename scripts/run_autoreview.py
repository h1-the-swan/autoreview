# -*- coding: utf-8 -*-

DESCRIPTION = """Given a list of paper IDs, output a directory with three pickle files, 
representing the seed papers split off from the original set, 
the remaining target papers, and the total set of candidate papers 
from collecting out- and in-citations 2 degrees out."""

import sys, os, time
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

from autoreview import Autoreview


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


def main(args):
    id_list = get_id_list(args.id_list)
    use_spark = not args.no_spark
    a = Autoreview(id_list=id_list,
                    citations=args.citations,
                    papers=args.papers,
                    outdir=args.outdir,
                    sample_size=args.sample_size,
                    random_seed=args.random_seed,
                    id_colname=args.id_colname,
                    cited_colname=args.cited_colname,
                    use_spark=use_spark)
    a.run()

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--id-list", help="list of ids for the seed papers (newline separated file)")
    parser.add_argument("--citations", help="citations data (to be read by spark)")
    parser.add_argument("--papers", help="papers/cluster data (to be read by spark)")
    parser.add_argument("--outdir", help="output directory (will be created)")
    parser.add_argument("--sample-size", type=int, default=200, help="number of articles to sample from the set to use to train the model (integer, default: 200)")
    parser.add_argument("--random-seed", type=int, default=999, help="random seed (integer, default: 999)")
    parser.add_argument("--id-colname", default='UID', help="column name for paper id (default: \"UID\")")
    parser.add_argument("--cited-colname", default='cited_UID', help="column name for cited paper id (default: \"cited_UID\")")
    parser.add_argument("--no-spark", action='store_true', help="don't use spark to collect candidate papers")
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
