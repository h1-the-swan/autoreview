import sys, os, time, pickle
from datetime import datetime
from timeit import default_timer as timer
from six import string_types
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv('admin.env')

# from db_connect_mag import Session, Paper, PaperAuthorAffiliation
# from db_connect_mag import db
from mysql_connect import get_db_connection
db = get_db_connection('mag_20180329')

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

from get_papers_2_degrees_out import _get_papers_and_save_from_id_list

def get_id_list_from_file(fname, header=True):
    id_list = []
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 and header is True:
                continue
            id_list.append(line.strip())
    return id_list

def main(args):

    outdir = os.path.abspath(args.outdir)

    paper_ids = get_id_list_from_file(args.id_list)

    _get_papers_and_save_from_id_list(paper_ids, args.outdir, args.num_seed_papers, args.seed)


if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="given a set of papers (i.e., references from a review article), take a random subset (i.e., the seed papers), and collect all of the citing and cited papers. Then add in the citing and cited papers for all of those papers.")
    parser.add_argument("id_list", help="file with newline-separated paper ids")
    parser.add_argument("-o", "--outdir", default="./", help="directory for output")
    parser.add_argument("--seed", type=int, default=999, help="random seed")
    parser.add_argument("--num-seed-papers", type=int, default=50, help="number of papers from the review paper's references to use as 'seed papers'. The rest will be used as 'target papers' (i.e., the 'needle' we want to find in the 'haystack')")
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




