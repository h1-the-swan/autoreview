# -*- coding: utf-8 -*-

DESCRIPTION = """Rerun prediction for a model and save all predictions as pickle"""

import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import pandas as pd
import numpy as np

sys.path.append('..')
from autoreview.util import get_best_model_from_datadir, prepare_data_for_model, predict_ranks_from_data

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

def main(args):
    logger.debug("getting best model from {}".format(args.dirname))
    pipeline = get_best_model_from_datadir(args.dirname)
    this_start = timer()

    logger.debug("preparing data for model...")
    test_papers, seed_papers, train_papers = prepare_data_for_model(args.dirname, year=2019)
    logger.debug("done preparing data. took {}".format(format_timespan(timer()-this_start)))

    logger.debug("predicting ranks from data...")
    this_start = timer()
    predictions = predict_ranks_from_data(pipeline, test_papers)
    logger.debug("done predicting. took {}".format(format_timespan(timer()-this_start)))

    outfname = os.path.join(args.dirname, 'predictions_all_no_metadata.pickle')
    logger.debug("writing to {}...".format(outfname))
    _predictions = predictions[['Paper_ID', 'pred_ranks']]
    _predictions.to_pickle(outfname)

    outfname = os.path.join(args.dirname, 'predictions_top.pickle')
    n = 1000000
    logger.debug("writing top {} to {}...".format(n, outfname))
    _predictions = predictions.head(n)
    _predictions.to_pickle(outfname)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("dirname", help="directory with model and paper data")
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
