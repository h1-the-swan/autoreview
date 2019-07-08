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

import pandas as pd

from dotenv import load_dotenv
load_dotenv('admin.env')
from mysql_connect import get_db_connection
db = get_db_connection('mag_20180329')

def get_top_ef_in_result(df):
    df = df.sort_values('pagerank', ascending=False)
    if df.iloc[0].pagerank > df.iloc[1].pagerank:
        return df.iloc[0]
    else:
        # give up
        return None

def query_mag_doi(doi):
    tbl1 = db.tables['Papers']
    tbl2 = db.tables['twolevel_cluster_relaxmap']
    j = tbl1.join(tbl2, tbl1.c.Paper_ID==tbl2.c.Paper_ID)
    sq = j.select(tbl1.c.DOI==doi)
    r = db.read_sql(sq)
    if len(r) == 0:
        return None, False
    elif len(r) == 1:
        return r.iloc[0], False
    else:
        return get_top_ef_in_result(r), True

def output_row(outf, wos_item, r, multiple_match_flag):
    if multiple_match_flag is True:
        multiple_match_flag = 1
    else:
        multiple_match_flag = 0
    row = [wos_item.name,
           str(r.Paper_ID.iloc[0]),  # since there are two 'Paper_ID' fields because of the join
           wos_item.doi,
           str(wos_item.pub_date),
           wos_item.title,
           wos_item.title_source,
           str(wos_item.num_citations),
           str(r.date),
           r.title,
           str(r.pagerank),
           str(multiple_match_flag)]
    outf.write("\t".join(row))
    outf.write("\n")


def main(args):
    df = pd.read_table(args.input, index_col=0)
    subset = df[df.doi.notnull()]
    subset = subset[subset.num_citations>=args.min_citations].sort_values('num_citations', ascending=False)
    logger.debug("{} papers found with DOI and num_citations above minimum threshold of {}".format(len(subset), args.min_citations))
    logger.debug("writing output to {}".format(args.output))
    with open(args.output, 'w') as outf:
        header = ['wos_id', 
                  'mag_id', 
                  'doi', 
                  'wos_date', 
                  'wos_title', 
                  'wos_title_source', 
                  'wos_num_citations', 
                  'mag_date', 
                  'mag_title', 
                  'mag_pagerank',
                  'multiple_match_flag']
        outf.write("\t".join(header))
        outf.write("\n")
        i = 0
        rows_written = 0
        for _, wos_item in subset.iterrows():
            r, multiple_match_flag = query_mag_doi(wos_item.doi)
            if r is not None:
                output_row(outf, wos_item, r, multiple_match_flag)
                rows_written += 1
            if i in [1,2,5,10,50,100] or i % 1000 == 0:
                logger.debug("i=={}. {} rows written.".format(i, rows_written))
            i += 1

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="match wos doi to mag")
    parser.add_argument("input", help="filename for wos reviews data (TSV)")
    parser.add_argument("-o", "--output", help="filename for output (TSV)")
    parser.add_argument("--min-citations", type=int, default=200, help="minimum citation count")
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
