# -*- coding: utf-8 -*-

DESCRIPTION = """Collect precision-recall values from best models (using spark)"""

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
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StructType, DoubleType, StringType, ArrayType

from ..config import Config
from ..util import get_best_model_from_datadir, prepare_data_for_model, get_best_model_path

DATA_BASEDIR = '..'
sys.path.append(DATA_BASEDIR)

SCHEMA = StructType([
    StructField("data_dir", StringType(), False),
    StructField("classifier_str", StringType(), True),
    StructField("average_precision", DoubleType(), True),
    StructField("precision", ArrayType(DoubleType()), True),
    StructField("recall", ArrayType(DoubleType()), True),
    StructField("threshold", ArrayType(DoubleType()), True),
    StructField("no_skill", DoubleType(), True),
])

def analyze_one_row(data_dir):
    """Analyze one row (for one model/data_dir)

    :data_dir: TODO
    :returns: TODO

    """
    test_papers, seed_papers, target_papers = prepare_data_for_model(data_dir)
    try:
        pipeline = get_best_model_from_datadir(data_dir)
    except FileNotFoundError:
        return empty_row(data_dir)
    X = test_papers.reset_index()
    y = X['target']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
    # y_score = pipeline.predict_proba(X_test)[:, 1]  # This is only predicting from X_test. We could also predict from all of X
    # average_precision = average_precision_score(y_test, y_score)
    # precision, recall, threshold = precision_recall_curve(y_test, y_score)
    # no_skill = (y==True).sum() / len(y)

    y_score = pipeline.predict_proba(X)[:, 1]
    average_precision = average_precision_score(y, y_score)
    precision, recall, threshold = precision_recall_curve(y, y_score)
    no_skill = (y==True).sum() / len(y)
    return_data = [{
        'data_dir': data_dir,
        'classifier_str': pipeline._final_estimator.__repr__(),
        'average_precision': average_precision,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
        'no_skill': no_skill,
    }]
    return pd.DataFrame(return_data)

def empty_row(data_dir):
    empty_row_data = [{
        'data_dir': data_dir,
        'classifier_str': None,
        'average_precision': None,
        'precision': None,
        'recall': None,
        'threshold': None,
        'no_skill': None,
    }]
    return pd.DataFrame(empty_row_data)


class Collector(object):

    """Load best models and data, and collect precision recall scores (using spark)"""

    def __init__(self, config):
        self.config = config
        self.spark = config.spark
        self.db = config._get_mysql_connection(db_name='jp_autoreview')

    @staticmethod
    def udf_data_collect(pdf):
        """To be used as a vectorized spark UDF (user-defined function)

        """
        sys.path.append(DATA_BASEDIR)
        data_dir = pdf['datadir'].iloc[0]
        return analyze_one_row(data_dir)

    def get_data_dirs(self):
        """Get all of the top performing models (for each review paper) 
        from MySQL
        :returns: pandas dataframe, one row per model/data_dir

        """
        df = pd.read_sql(self.db.tables['pipeline_tests'].select(), self.db.engine)
        df.set_index('id', inplace=True)
        top_results = df.sort_values('score_correctly_predicted', ascending=False).drop_duplicates(['review_paper_id', 'random_seed'])
        return top_results


    def test(self, args):
        """test
        """
        top_results = self.get_data_dirs()
        data_dir = top_results.datadir.iloc[:100]
        data_dir = data_dir.apply(lambda x: os.path.join(DATA_BASEDIR, x))
        self.udf_data_collect = pandas_udf(self.udf_data_collect, SCHEMA, PandasUDFType.GROUPED_MAP)
        sdf = self.spark.createDataFrame(data_dir.to_frame())
        res = sdf.groupby('datadir').apply(self.udf_data_collect)
        print(res)
        res.write.parquet('test_parquet')

        # dfs_res = []
        # for dirname in data_dir:
        #     dfs_res.append(analyze_one_row(dirname))
        # df = pd.concat(dfs_res)
        # df.to_parquet('test2_parquet')





    def main(self, args):
        top_results = self.get_data_dirs()
        data_dir = top_results.datadir
        logger.debug("analyzing results for {} models...".format(len(data_dir)))
        data_dir = data_dir.apply(lambda x: os.path.join(DATA_BASEDIR, x))
        self.udf_data_collect = pandas_udf(self.udf_data_collect, SCHEMA, PandasUDFType.GROUPED_MAP)
        sdf = self.spark.createDataFrame(data_dir.to_frame())
        res = sdf.groupby('datadir').apply(self.udf_data_collect)
        res.write.parquet(args.output)
        

def main(args):
    try:
        config = Config()
        collector = Collector(config)
        collector.main(args)
    finally:
        config.teardown()

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("output", help="output (parquet)")
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
