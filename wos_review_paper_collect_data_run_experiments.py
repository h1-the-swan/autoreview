import sys, os, time, json
import subprocess
import shlex
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

import pandas as pd
import numpy as np

from dotenv import load_dotenv
logger.debug("loading dotenv...")
success_load_dotenv = load_dotenv('admin.env')
if success_load_dotenv:
    logger.debug("dotenv loaded successfully.")
else:
    logger.warn("failed to load dotenv")
from mysql_connect import get_db_connection
db_mag = get_db_connection('mag_2017-10')

from utils.autoreview_utils import prepare_directory

WOS_DATA = 'data/wos_papers_reviews_with_citation_count.tsv'
COLLECT_DATA_AND_RUN_EXPERIMENTS_SCRIPT = 'get_haystack_then_run_experiments.py'

def mag_query_doi(doi, id_colname='Paper_ID'):
    tbl1 = db_mag.tables['Papers']
    tbl2 = db_mag.tables['rank']
    j = tbl1.join(tbl2, tbl1.c[id_colname]==tbl2.c[id_colname])
    sq = j.select(tbl1.c.DOI==doi)
    df = db_mag.read_sql(sq)
    return df

def get_mag_result_for_doi(doi):
    # query for DOI. If there is more than one result, use the highest ranked one (by Eigenfactor)
    df = mag_query_doi(doi)
    if len(df) == 0:
        return None
    elif len(df) == 1:
        return df.iloc[0]
    else:
        try:
            return df.sort_values('EF', ascending=False).iloc[0]
        except:
            return None

def main(args):
    wos_papers = pd.read_table(WOS_DATA, index_col=0)
    wos_item = wos_papers.loc[args.wosid]
    mag_result = get_mag_result_for_doi(wos_item.doi)
    if mag_result is None:
        logger.info("no entry found in the MAG database for DOI {}. Exiting...".format(wos_item.doi))
        return

    dirname = prepare_directory(mag_result.Paper_ID, args.description)
    cmd = "python {} {} --year {} --min-seed={} --max-seed={} --num-seed-papers {} --description {} --debug".format(COLLECT_DATA_AND_RUN_EXPERIMENTS_SCRIPT, 
            mag_result.Paper_ID, 
            mag_result.year, 
            args.min_seed, 
            args.max_seed, 
            args.num_seed_papers, 
            args.description)
    logger.debug("running command: {}...".format(cmd))
    cmd_list = shlex.split(cmd)
    log_fname = "collect_seed_{:03d}_{:%Y%m%d}.log".format(random_seed, datetime.now())
    log_fname = os.path.join(dirname, log_fname)
    logger.debug("logging output to {}".format(log_fname))
    start = timer()
    with open(log_fname, 'w') as logf:
        process = subprocess.Popen(cmd_list, stdout=logf, stderr=logf)
        process.wait()  # wait for the process to finish

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="given a WOS ID for a review paper (with DOI), look up the paper in the MAG database using DOI, collect data for a range of random seeds, and run experiments")
    parser.add_argument("wosid", help="WOS paper id for the review article")
    parser.add_argument("--min-seed", type=int, default=1, help="This script will collect data and run experiments for every integer seed value between min-seed and max-seed. default range is [1-6)")
    parser.add_argument("--max-seed", type=int, default=6, help="max random seed (non-inclusive). see help for --min-seed")
    parser.add_argument("--num-seed-papers", type=int, default=50, help="number of papers from the review paper's references to use as 'seed papers'. The rest will be used as 'target papers' (i.e., the 'needle' we want to find in the 'haystack')")
    parser.add_argument("--description", help="text that will be appended to the directory name")
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
