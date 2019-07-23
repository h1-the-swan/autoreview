# -*- coding: utf-8 -*-

DESCRIPTION = """Remove the seed papers from the predictions"""

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
from autoreview.util import prepare_directory

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

def main(args):
    df_predictions = pd.read_pickle(args.predictions).reset_index()
    dfs = []
    for dirname in args.dirnames:
        dfs.append(pd.read_pickle(os.path.join(dirname, 'seed_papers.pickle')))
        dfs.append(pd.read_pickle(os.path.join(dirname, 'target_papers.pickle')))
    all_dfs = pd.concat(dfs)
    all_dfs_dedup = all_dfs.drop_duplicates()

    df_predictions = df_predictions[~df_predictions.Paper_ID.isin(all_dfs_dedup.Paper_ID)]

    outfname = os.path.join(args.outdir, 'all_predictions_combined_rm_origset.pickle')
    logger.debug("writing to {}".format(outfname))
    df_predictions.to_pickle(outfname)

    outfname = os.path.join(args.outdir, 'top_predictions_combined_rm_origset.tsv')
    logger.debug("writing to {}".format(outfname))
    df_predictions.head(1000).to_csv(outfname, sep='\t')

        

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("predictions", help="pickle file with predictions dataframe (ordered descending by score)")
    parser.add_argument("--dirnames", nargs='+', help="directories with seed and target pickled dataframes")
    parser.add_argument("--outdir", help="output directory to store the predictions (must already exist)")
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


