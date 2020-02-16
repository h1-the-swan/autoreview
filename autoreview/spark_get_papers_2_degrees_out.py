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
from sklearn.model_selection import train_test_split

sys.path.append('..')
from util import load_spark_session
spark = load_spark_session(appName="spark_get_papers_2_degrees_out", envfile='../spark.env')

import logging
logger = logging.getLogger(__name__)

def load_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state

def combine_ids(sdfs):
    """Given a list of spark dataframes with columns ['UID', 'cited_UID']
    return a dataframe with one column 'UID' containing all of the UIDs in both columns of the input dataframes
    """
    sdf_combined = spark.createDataFrame([], schema='UID string')
    for sdf in sdfs:
        # add 'UID' column
        sdf_combined = sdf_combined.union(sdf.select(['UID']))
        
        # add 'cited_UID' column (need to rename)
        sdf_combined = sdf_combined.union(sdf.select(['cited_UID']).withColumnRenamed('cited_UID', 'UID'))
    return sdf_combined.drop_duplicates()

def get_year(x):
    if x:
        return x.year
    else:
        return None

def save_pandas_dataframe_to_pickle(df, outfname):
    df['year'] = df['pub_date'].apply(get_year)
    columns_rename = {
        'UID': 'Paper_ID',
        'flow': 'EF'
    }
    df.rename(columns=columns_rename, inplace=True)
    df.to_pickle(outfname)

def main(args):
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise RuntimeError("output path {} already exists!".format(outdir))
    os.mkdir(outdir)

    random_state = load_random_state(args.random_seed)

    sdf_papers = spark.read.parquet(args.papers)

    df_nas2 = pd.read_csv(args.id_list, sep='\t')
    seed_papers, target_papers = train_test_split(df_nas2, train_size=args.sample_size, random_state=random_state)
    logger.debug("getting citing and cited papers for a set of {} papers".format(len(seed_papers)))
    sdf_seed = spark.createDataFrame(seed_papers[['UID']])
    sdf_target = spark.createDataFrame(target_papers[['UID']])

    outfname = os.path.join(outdir, 'seed_papers.pickle')
    logger.debug('saving seed papers to {}'.format(outfname))
    start = timer()
    df_seed = sdf_seed.join(sdf_papers, on='UID', how='inner').toPandas()
    save_pandas_dataframe_to_pickle(df_seed, outfname)
    logger.debug("done saving seed papers. took {}".format(format_timespan(timer()-start)))

    outfname = os.path.join(outdir, 'target_papers.pickle')
    logger.debug('saving target papers to {}'.format(outfname))
    start = timer()
    df_target = sdf_seed.join(sdf_papers, on='UID', how='inner').toPandas()
    save_pandas_dataframe_to_pickle(df_target, outfname)
    logger.debug("done saving target papers. took {}".format(format_timespan(timer()-start)))

    sdf_citations = spark.read.csv(args.citations, sep='\t', header=True)

    sdf_outcitations1 = sdf_citations.join(sdf_seed, on='UID', how='inner')

    _sdf_renamed = sdf_seed.withColumnRenamed('UID', '_UID')
    sdf_incitations1 = sdf_citations.join(_sdf_renamed, on=sdf_citations['UID']==_sdf_renamed['_UID'])
    sdf_incitations1 = sdf_incitations1.drop('_UID')

    sdf_combined = combine_ids([sdf_incitations1, sdf_outcitations1])

    # do it all again to get second degree

    logger.debug("collecting 2nd degree...")
    sdf_outcitations2 = sdf_citations.join(sdf_combined, on='UID', how='inner')

    _sdf_renamed = sdf_combined.withColumnRenamed('UID', '_UID')
    sdf_incitations2 = sdf_citations.join(_sdf_renamed, on=sdf_citations['UID']==_sdf_renamed['_UID'])
    sdf_incitations2 = sdf_incitations2.drop('_UID')

    sdf_combined = combine_ids([sdf_incitations2, sdf_outcitations2])

    outfname = os.path.join(outdir, 'test_papers.pickle')
    logger.debug('saving test papers to {}'.format(outfname))
    start = timer()
    df_combined = sdf_combined.join(sdf_papers, on='UID', how='inner').toPandas()
    save_pandas_dataframe_to_pickle(df_combined, outfname)
    logger.debug("done saving test papers. took {}".format(format_timespan(timer()-start)))

    spark.stop()

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="Using Apache Spark, get papers two degrees out from a sample of the seed set")
    parser.add_argument("--id-list", help="list of ids for the seed papers")
    parser.add_argument("--citations", help="citations TSV file (or directory for spark csv files)")
    parser.add_argument("--papers", help="directory for parquet files with papers/cluster data")
    parser.add_argument("--outdir", help="output directory (will be created)")
    parser.add_argument("--sample-size", type=int, default=200, help="number of articles to sample from the set to use to train the model (integer, default: 200)")
    parser.add_argument("--random-seed", type=int, default=999, help="random seed (integer, default: 999)")
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
