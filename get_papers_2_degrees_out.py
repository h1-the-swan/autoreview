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

def parse_id(id_to_parse=None):
    # Should take ID in any form (str, int, list of int/str, pandas series) and return a list of IDs as strings
    if isinstance(id_to_parse, string_types):
        idlist = [id_to_parse]
        return idlist
    else:
        try:
            idlist = list(id_to_parse)
            idlist = [str(item) for item in idlist]
            return idlist
        except TypeError:
            idlist = [str(int(id_to_parse))]
            return idlist
    return id_to_parse


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

def collect_citing_and_cited(papers, collected=[], collected_ids=set(), counter=Counter()):
    ids_to_get = set()
    for i, paper in enumerate(papers):
        citing_paper_ids = get_reference_ids(paper['Paper_ID'], direction='out', tablename='PaperReferences_out')
        for pid in citing_paper_ids:
            counter[pid] += 1
            if pid not in collected_ids:
                ids_to_get.add(pid)
        cited_paper_ids = get_reference_ids(paper['Paper_ID'], direction='in', tablename='PaperReferences_in')
        for pid in cited_paper_ids:
            counter[pid] += 1
            if pid not in collected_ids:
                ids_to_get.add(pid)
    logger.debug("querying for {} papers...".format(len(ids_to_get)))
    collected.extend(get_papers(list(ids_to_get)))

    return collected, collected_ids, counter

def pickle_dataframe_from_papers(papers, outfname):
    rows = []
    for p in papers:
        rows.append({
            'Paper_ID': p['Paper_ID'],
            'title': p['title'],
            'year': p['year'],
            'EF': p['EF'],
            'cl': p['cl']
        })
    df = pd.DataFrame(rows)
    df.to_pickle(outfname)

def get_reference_ids(paper_id, direction='out', tablename='PaperReferences', colnames=['Paper_ID', 'Paper_reference_ID']):
    if direction.lower().startswith('in'):
        colnames = [colnames[1], colnames[0]]
    tbl = db.tables[tablename]
    sq = tbl.select(tbl.c[colnames[0]]==paper_id)
    result = db.engine.execute(sq).fetchall()
    return [x[colnames[1]] for x in result]

def get_paper_as_dict(paper):
    tbl_papers = db.tables['Papers']
    # tbl_rank = db.tables['rank']
    tbl_rank = db.tables['twolevel_cluster_relaxmap']
    tbl_tree = db.tables['tree']
    paper_dict = {
        'Paper_ID': paper[tbl_papers.c['Paper_ID']],
        'title': paper[tbl_papers.c['title']],
        'year': paper[tbl_papers.c['year']],
        'EF': paper[tbl_rank.c['pagerank']],
        'cl': paper[tbl_tree.c['cl']]
    }
    return paper_dict

def get_papers(paper_id_list, tablenames=['Papers', 'twolevel_cluster_relaxmap', 'tree'], id_colname='Paper_ID'):
    tbls = [db.tables[tablename] for tablename in tablenames]
    first_tbl = tbls[0]
    j = first_tbl
    if len(tbls) > 1:
        for tbl in tbls[1:]:
            j = j.join(tbl, first_tbl.c[id_colname]==tbl.c[id_colname])
    # split id list into serial queries
    i = 0
    step = 10000
    all_results = []
    while True:
        id_subset = paper_id_list[i:i+step]
        if not id_subset:
            break
        sq = j.select(first_tbl.c[id_colname].in_(id_subset))
        result = db.engine.execute(sq).fetchall()
        all_results.extend([get_paper_as_dict(x) for x in result])
        i = i + step
    return all_results

def get_papers_and_save(review_paper_id, outdir='./', num_seed_papers=50, random_state=999):
    if isinstance(random_state, np.random.RandomState):
        pass
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        raise RuntimeError('argument random_state must be type np.random.RandomState or integer')

    review_paper = get_papers(parse_id(review_paper_id))[0]
    logger.debug("getting references from paper {} (Paper_ID {})".format(review_paper['title'], review_paper_id))
    start = timer()
    reference_ids = get_reference_ids(review_paper_id, direction='out', tablename='PaperReferences_out')
    reference_papers = get_papers(reference_ids)
    logger.debug("{} references found in {}".format(len(reference_papers), format_timespan(timer()-start)))
    seed_papers, target_papers = train_test_split(reference_papers, train_size=num_seed_papers, random_state=random_state)

    outfname = os.path.join(outdir, "seed_papers.pickle")
    logger.debug("saving {} papers to {}".format(len(seed_papers), outfname))
    pickle_dataframe_from_papers(seed_papers, outfname)

    # target_papers = set(target_papers)
    target_paper_ids = set([p['Paper_ID'] for p in target_papers])

    outfname = os.path.join(outdir, "target_papers.pickle")
    logger.debug("saving {} papers to {}".format(len(target_papers), outfname))
    pickle_dataframe_from_papers(target_papers, outfname)



    logger.debug("getting citing and cited papers for a sample of {} papers...".format(len(seed_papers)))

    start = timer()
    test_papers = []
    test_paper_ids = set()
    c = Counter()
    cur_papers = seed_papers
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



    start = timer()
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

def main(args):

    outdir = os.path.abspath(args.outdir)

    review_paper_id = args.id

    get_papers_and_save(review_paper_id, outdir, args.num_seed_papers, args.seed)


if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="given a set of papers (i.e., references from a review article), take a random subset (i.e., the seed papers), and collect all of the citing and cited papers. Then add in the citing and cited papers for all of those papers.")
    parser.add_argument("id", help="paper id for the review article that contains the references")
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



