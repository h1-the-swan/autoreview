# -*- coding: utf-8 -*-

DESCRIPTION = """Train classifiers for autoreview.
Use this in the case where seed and candidate paper sets have already been collected and saved as pickle files."""

import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

from autoreview import Autoreview
from autoreview.util import load_data_from_pickles


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
    a = Autoreview(args.outdir, args.random_seed)
    candidate_papers, seed_papers, target_papers = load_data_from_pickles(a.outdir)
    a.train_models(seed_papers=seed_papers, target_papers=target_papers, candidate_papers=candidate_papers)

if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--outdir", help="output directory (will be created)")
    parser.add_argument("--random-seed", type=int, default=999, help="random seed (integer, default: 999)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        root_logger.setLevel(logging.INFO)
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))


