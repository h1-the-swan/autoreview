import sys, os, time
from glob import glob
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

from sklearn.externals import joblib

from dotenv import load_dotenv
load_dotenv('admin.env')

from mysql_connect import get_db_connection
db = get_db_connection('jp_autoreview')
from sqlalchemy.orm import sessionmaker
from models.jp_autoreview import PipelineTest
Session = sessionmaker(bind=db.engine)

def get_feature_names(pipeline):
    u = pipeline.named_steps['union']
    features = []
    for item in u.transformer_list:
        features.append(item[0])
    return features

def main(args):
    df = db.read_sql(db.tables['pipeline_tests'].select())
    all_datadirs = df.datadir.unique().tolist()
    for datadir in all_datadirs:
        g = glob(os.path.join(datadir, 'best_model*'))
        for model_path in g:
            fname = os.path.join(model_path, 'best_model.pickle')
            if os.path.exists(fname):
                pipeline = joblib.load(fname)
                features = str(get_feature_names(pipeline))
                session = Session()
                r = session.query(PipelineTest) \
                        .filter_by(datadir=datadir) \
                        .filter_by(clf=str(pipeline._final_estimator)) \
                        .filter_by(features=features) \
                        .all()
                if len(r) == 1:
                    pt = r[0]
                    pt.saved_model = fname
                    logger.debug("adding model path {} to database record (id: {})".format(fname, pt.id))
                    session.add(pt)
                    session.commit()
                elif len(r) == 0:
                    logger.error("no database record found for saved model {}".format(fname))
                else:
                    logger.error("multiple ({}) database records found for saved model {}".format(len(r), fname))

                session.close()


if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="identify saved models")
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
