# -*- coding: utf-8 -*-

DESCRIPTION = """Reload model and make predictions"""

import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

from sklearn.externals import joblib
from utils.autoreview_utils import get_best_model_path, prepare_data_for_model, predict_ranks_from_data

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

def write_output(df, outfname, N=None):
    """Write dataframe to pickle

    :df: pandas dataframe with predictions (sorted by probability, descending)
    :outfname: output filename (pickle)
    :N: only write top(N) results

    """
    _df = df.copy()
    if N is not None:
        _df = _df.head(N)
    _df.to_pickle(outfname, compression=None)

def main(args):
    outfname = args.output
    if not outfname:
        outfname = os.path.join(args.datadir, 'predictions.pickle')
    model_fname = get_best_model_path(args.datadir)
    logger.debug("using model: {}".format(model_fname))
    logger.debug("loading model...")
    pipeline = joblib.load(model_fname)
    logger.debug("preparing data...")
    test_papers, seed_papers, train_papers = prepare_data_for_model(args.datadir, year=2018)
    logger.debug("predicting...")
    df_predictions = predict_ranks_from_data(pipeline, test_papers)
    df_predictions = df_predictions[df_predictions.target==False].drop(columns='target')
    logger.debug("writing to {}".format(outfname))
    write_output(df_predictions, outfname, args.num_predictions)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("datadir", help="data directory with seed/target/test pickles and best model directory")
    parser.add_argument("-o", "--output", help="output filename (pickled pandas dataframe). will use `predictions.pickle` if not specified). The output file will be written to `datadir`.")
    parser.add_argument("-n", "--num-predictions", type=int, default=1000, help="number of predictions to include (default: 1000)")
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
