import sys, os, time, json, pickle
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

from dotenv import load_dotenv
load_dotenv('admin.env')

from h1theswan_utils.microsoft_academic_api import EvaluateQuery, convert_inverted_abstract_to_abstract_words

import pandas as pd



class CollectAbstracts(object):

    """Collect abstracts from the microsoft academic api"""

    def __init__(self, paper_ids, output_dir):
        self.paper_ids = paper_ids
        self.output_dir = os.path.abspath(output_dir)

        self.inverted_abstracts = {}
        self.list_abstracts = {}
        self.failed = []  # list of paper ids for which we couldn't get the abstract

    @staticmethod
    def query_abstract(paper_id):
        eq = EvaluateQuery("Id={}".format(paper_id))
        eq.attributes = 'E'
        r = eq.get()
        ents = r.get('entities')
        if ents:
            ent = ents[0]
            E = ent.get('E')
            if E:
                E = json.loads(E)
                iabs = E.get('IA')
                if iabs:
                    return iabs
        # if we reach here, the api query did not return an abstract
        return None

    def process_one_paper(self, paper_id):
        """get inverted abstract and list abstract for one paper


        """
        iabs = self.query_abstract(paper_id)
        if iabs is None:
            self.failed.append(paper_id)
            return False

        inverted_index = iabs['InvertedIndex']
        index_length = iabs['IndexLength']

        self.inverted_abstracts[paper_id] = inverted_index
        try:
            self.list_abstracts[paper_id] = convert_inverted_abstract_to_abstract_words(inverted_index, index_length)
        except:
            logger.error("error encountered for paper_id {}".format(paper_id))
            raise
        return True

    def pickle_on_fail(self):
        """pickle data when an exception is encountered

        """
        logger.error("exception encountered! saving data before exiting...")
        outfname = os.path.join(self.output_dir, 'INTERRUPTED_inverted_abstracts.pickle')
        logger.debug("saving {} inverted abstracts to {}...".format(len(self.inverted_abstracts), outfname))
        with open(outfname, 'wb') as outf:
            pickle.dump(self.inverted_abstracts, outf)
        logger.debug("done.")
                
        outfname = os.path.join(self.output_dir, 'INTERRUPTED_list_abstracts.pickle')
        logger.debug("saving {} abstracts (lists of words) to {}...".format(len(self.list_abstracts), outfname))
        with open(outfname, 'wb') as outf:
            pickle.dump(self.list_abstracts, outf)
        logger.debug("done.")

        outfname = os.path.join(self.output_dir, 'INTERRUPTED_failed_paperids.txt')
        logger.debug("saving {} failed paper ids to {}...".format(len(self.failed), outfname))
        with open(outfname, 'w') as outf:
            for paper_id in self.failed:
                outf.write("{}\n".format(paper_id))
        logger.debug("done.")

    def process_all(self):
        start = timer()
        query_times = []
        i = 0
        skipped = 0
        skip_ids = set.union(set(self.inverted_abstracts.keys()), set(self.failed))
        for paper_id in self.paper_ids:
            # if paper_id in self.inverted_abstracts.keys() or paper_id in self.failed:
            if paper_id in skip_ids:
                # if it's already there, then skip it
                skipped += 1
                continue
            q_start = timer()
            self.process_one_paper(paper_id)
            query_times.append(timer()-q_start)
            i += 1
            if i in [1,2,5,10,50,100,500,1000,10000] or i % 50000 == 0:
                logger.debug("{} paper_ids processed. {} abstracts collected. {} failed. {} skipped. avg time per query: {:.02f} seconds".format(i, len(self.inverted_abstracts), len(self.failed), skipped, (sum(query_times))/len(query_times)))

        logger.debug("done with {} queries. took {}".format(len(query_times), format_timespan(timer()-start)))
        logger.debug("found {} abstracts. {} failed. {} skipped".format(len(self.inverted_abstracts), len(self.failed), skipped))

    def run(self):
        if not self.paper_ids:
            raise RuntimeError("No paper ids specified")
        start = timer()
        try:
            self.process_all()
        except Exception as e:
            try:
                time_to_wait = 1200  # 20 minutes
                logger.warn(repr(e))
                self.pickle_on_fail()
                logger.warn("------RESTARTING QUERY LOOP---------")
                logger.warn("waiting {} seconds and then restarting query loop...".format(time_to_wait))
                logger.warn("")
                time.sleep(time_to_wait)
                self.process_all()
            except:
                # give up
                self.pickle_on_fail()
                raise

        logger.debug("\ntotal runtime (including restarts): {}".format(format_timespan(timer()-start)))

        outfname = os.path.join(self.output_dir, 'inverted_abstracts.pickle')
        logger.debug("saving {} inverted abstracts to {}...".format(len(self.inverted_abstracts), outfname))
        with open(outfname, 'wb') as outf:
            pickle.dump(self.inverted_abstracts, outf)
        logger.debug("done.")
                
        outfname = os.path.join(self.output_dir, 'list_abstracts.pickle')
        logger.debug("saving {} abstracts (lists of words) to {}...".format(len(self.list_abstracts), outfname))
        with open(outfname, 'wb') as outf:
            pickle.dump(self.list_abstracts, outf)
        logger.debug("done.")

        outfname = os.path.join(self.output_dir, 'failed_paperids.txt')
        logger.debug("saving {} failed paper ids to {}...".format(len(self.failed), outfname))
        with open(outfname, 'w') as outf:
            for paper_id in self.failed:
                outf.write("{}\n".format(paper_id))
        logger.debug("done.")
        

def main(args):
    # test
    # train_papers_df = pd.read_pickle('data/collect_haystack_20180409_2/train_papers.pickle')
    # train_pids = train_papers_df.Paper_ID.tolist()
    # output_dir = 'data/collect_haystack_20180409_2/test_abstracts'

    output_dir = os.path.abspath(args.outdir)
    logger.debug("output directory will be: {}".format(output_dir))

    logger.debug("getting paper ids...")
    paper_ids = []
    with open(args.pids, 'r') as f:
        for line in f:
            line = line.strip()
            paper_ids.append(line)
    logger.debug("found {} paper ids.".format(len(paper_ids)))

    collect = CollectAbstracts(paper_ids, output_dir)
    if args.checkpoint_inverted is not None:
        with open(args.checkpoint_inverted, 'rb') as f:
            collect.inverted_abstracts = pickle.load(f)
        logger.debug("{} inverted abstracts loaded from checkpoint_inverted".format(len(collect.inverted_abstracts)))
    if args.checkpoint_list is not None:
        with open(args.checkpoint_list, 'rb') as f:
            collect.list_abstracts = pickle.load(f)
        logger.debug("{} list abstracts loaded from checkpoint_list".format(len(collect.list_abstracts)))

    if args.skip is not None:
        with open(args.skip, 'r') as f:
            skip_ids = []
            for line in f:
                skip_ids.append(line.strip())
        collect.failed = skip_ids
        logger.debug("{} ids to skip loaded from {}".format(len(skip_ids), args.skip))
    collect.run()

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="get abstracts from microsoft API and save both inverted abstracts and list abstracts as pickles")
    parser.add_argument("pids", help="file with (newline-separated) list of paper ids")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--checkpoint-inverted", help="file containing already-collected inverted abstracts (from a previous interrupted run)")
    parser.add_argument("--checkpoint-list", help="file containing already-collected list abstracts (from a previous interrupted run)")
    parser.add_argument("--skip", help="text file containing (newline-separated) ids to skip")
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
