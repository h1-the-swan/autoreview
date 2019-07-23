# -*- coding: utf-8 -*-

DESCRIPTION = """combine predictions from multiple runs"""

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
    prepare_directory(args.outdir)
    dfs = []
    preds = {}
    for dirname in args.dirnames:
        fname = os.path.join(dirname, 'predictions_top.pickle')
        this_df = pd.read_pickle(fname).set_index('Paper_ID')
        dfs.append(this_df)
        for id_, p in this_df.pred_ranks.items():
            if id_ in preds:
                preds[id_] += p
            else:
                preds[id_] = p
    preds = pd.Series(preds).sort_values(ascending=False)
    all_papers = pd.concat(_df for _df in dfs)
    all_papers = all_papers.drop(columns='pred_ranks').drop_duplicates()
    all_papers['combined_preds'] = preds
    all_papers = all_papers.sort_values('combined_preds', ascending=False)

    outfname = os.path.join(args.outdir, 'all_predictions_combined.pickle')
    logger.debug("writing to {}".format(outfname))
    all_papers.to_pickle(outfname)

    outfname = os.path.join(args.outdir, 'top_predictions_combined.tsv')
    logger.debug("writing to {}".format(outfname))
    all_papers.head(1000).to_csv(outfname, sep='\t')

        

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("dirnames", nargs='+', help="directories with predictions")
    parser.add_argument("--outdir", help="output directory (will be created)")
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

