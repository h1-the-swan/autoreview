import sys, os, time, pickle
from datetime import datetime
from timeit import default_timer as timer
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

from db_connect_mag import Session, Paper, PaperAuthorAffiliation

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)


def tree_distance(n1, n2, sep=":"):
    # https://en.wikipedia.org/wiki/Lowest_common_ancestor
    # the distance from v to w can be computed as 
    # the distance from the root to v, plus the distance from 
    # the root to w, minus twice the distance from 
    # the root to their lowest common ancestor
    v, w = [n.split(sep) for n in [n1, n2]]
    distance_root_to_v = len(v)
    distance_root_to_w = len(w)
    
    distance_root_to_lca = 0
    for i in range(min(distance_root_to_v, distance_root_to_w)):
        if v[i] == w[i]:
            distance_root_to_lca += 1
        else:
            break
    return distance_root_to_v + distance_root_to_w - (2*distance_root_to_lca)
    



def distances_two_groups(g1, g2):
    distances = []
    for n1 in g1:
        for n2 in g2:
            if n1 == n2:
                continue
            distances.append(tree_distance(n1, n2))
    return distances

def collect_citing_and_cited(papers, collected=set(), collected_ids=set(), counter=Counter()):
    for i, paper in enumerate(papers):
        for pr in paper.paperrefs_citing:
            pid = pr.Paper_reference_ID
            counter[pid] += 1
            if pid not in collected_ids:
                collected_ids.add(pid)
                collected.add(pr.paper_cited)
        for pr in paper.paperrefs_cited:
            pid = pr.Paper_ID
            counter[pid] += 1
            if pid not in collected_ids:
                collected_ids.add(pid)
                collected.add(pr.paper_citing)
        if (i+1) in [1,2,5,10,20,50,100,200,500,1000,5000] or (i+1) % 10000 == 0:
            logger.debug("done with {} papers. len(collected)=={}".format(i+1, len(collected)))
    return collected, collected_ids, counter

def pickle_dataframe_from_papers(papers, outfname):
    rows = []
    for p in papers:
        rows.append({
            'Paper_ID': p.Paper_ID,
            'title': p.title,
            'year': p.year,
            'EF': p.EF,
            'cl': p.cl
        })
    df = pd.DataFrame(rows)
    df.to_pickle(outfname)


def main(args):
    session = Session()

    outdir = os.path.abspath(args.outdir)

    # review paper on community detection in graphs
    review_paper_id = 2127048411
    review_paper = session.query(Paper).get(review_paper_id)
    logger.debug("getting references from paper {} (Paper_ID {})".format(review_paper_id, review_paper.title))
    start = timer()
    papers = [pr.paper_cited for pr in review_paper.paperrefs_citing]
    logger.debug("{} references found in {}".format(len(papers), format_timespan(timer()-start)))
    train_papers, target_papers = train_test_split(papers, train_size=50, random_state=999)

    outfname = os.path.join(outdir, "train_papers.pickle")
    logger.debug("saving {} papers to {}".format(len(train_papers), outfname))
    pickle_dataframe_from_papers(train_papers, outfname)



    target_papers = set(target_papers)
    target_paper_ids = set([p.Paper_ID for p in target_papers])

    outfname = os.path.join(outdir, "target_papers.pickle")
    logger.debug("saving {} papers to {}".format(len(target_papers), outfname))
    pickle_dataframe_from_papers(target_papers, outfname)

    logger.debug("getting citing and cited papers for a sample of {} papers...".format(len(train_papers)))

    start = timer()
    test_papers = set()
    test_paper_ids = set()
    c = Counter()
    cur_papers = list(train_papers)
    test_papers, test_paper_ids, c = collect_citing_and_cited(cur_papers, test_papers, test_paper_ids, c)
    logger.debug("done collecting papers. len(test_papers)=={}. took {}".format(len(test_papers), format_timespan(timer()-start)))

    outfname = os.path.join(outdir, "counter_checkpoint.pickle")
    logger.debug("saving counter to {}".format(outfname))
    with open(outfname, "wb") as outf:
        pickle.dump(c, outf)

    start = timer()
    outfname = os.path.join(outdir, "test_papers_checkpoint.pickle")
    logger.debug("saving {} papers to {}".format(len(test_papers), outfname))
    pickle_dataframe_from_papers(test_papers, outfname)
    logger.debug("done saving. took {}".format(format_timespan(timer()-start)))


    logger.debug("getting citing and cited papers for {} papers...".format(len(test_papers)))
    cur_papers = list(test_papers)
    test_papers, test_paper_ids, c = collect_citing_and_cited(cur_papers, test_papers, test_paper_ids, c)
    logger.debug("done collecting papers. len(test_papers)=={}. took {}".format(len(test_papers), format_timespan(timer()-start)))

    outfname = os.path.join(outdir, "counter.pickle")
    logger.debug("saving counter to {}".format(outfname))
    with open(outfname, "wb") as outf:
        pickle.dump(c, outf)

    start = timer()
    outfname = os.path.join(outdir, "test_papers.pickle")
    logger.debug("saving {} papers to {}".format(len(test_papers), outfname))
    pickle_dataframe_from_papers(test_papers, outfname)
    logger.debug("done saving. took {}".format(format_timespan(timer()-start)))


    session.close()



if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="given a set of papers (i.e., references from a review article), take a random subset (i.e., the seed papers), and collect all of the citing and cited papers. Then add in the citing and cited papers for all of those papers.")
    parser.add_argument("-o", "--outdir", default="./", help="directory for output")
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



