import sys, os, time
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

from utils.autoreview_utils import prepare_directory

COLLECT_PAPERS_SCRIPT = 'get_papers_2_degrees_out.py'
EXPERIMENTS_SCRIPT = 'pipeline_experiments.py'

def main(args):
    if args.year is None:
        raise RuntimeError("must specify the --year of the review article")
    if args.min_seed >= args.max_seed:
        raise RuntimeError("--min-seed must be less than --max-seed (min_seed=={}, max_seed=={})".format(args.min_seed, args_max_seed))
    dirname = prepare_directory(args.paperid, args.description)

    for random_seed in range(args.min_seed, args.max_seed):
        seed_dirname = os.path.join(dirname, "seed_{:03d}".format(random_seed))
        if os.path.exists(seed_dirname):
            raise RuntimeError("directory {} already exists!".format(seed_dirname))
        logger.debug("creating directory: {}".format(seed_dirname))
        os.mkdir(seed_dirname)
        cmd = "python {} {} -o {} --seed {} --num-seed-papers {} --debug".format(COLLECT_PAPERS_SCRIPT, args.paperid, seed_dirname, random_seed, args.num_seed_papers)
        logger.debug("running command: {}...".format(cmd))
        cmd_list = shlex.split(cmd)
        log_fname = "collect_seed_{:03d}_{:%Y%m%d}.log".format(random_seed, datetime.now())
        log_fname = os.path.join(dirname, log_fname)
        logger.debug("logging output to {}".format(log_fname))
        start = timer()
        with open(log_fname, 'w') as logf:
            process = subprocess.Popen(cmd_list, stdout=logf, stderr=logf)
            process.wait()  # wait for the process to finish
        logger.debug("collected seed papers for random seed {} in {}".format(random_seed, format_timespan(timer()-start)))

        # now that the data have been collected for this review paper using this random seed,
        # run experiments (in the background).
        experiment_processes = []
        # run with two feature sets. first with just network features:
        cmd = "python {} {} --seed 999 --review-id {} --dataset-seed {} --year {} --save-best --debug".format(EXPERIMENTS_SCRIPT, seed_dirname, args.paperid, random_seed, args.year)
        logger.debug("running command (in the background): {}...".format(cmd))
        cmd_list = shlex.split(cmd)
        log_fname = "experiments_featuresAvgDistanceAndEF_{:%Y%m%d%H%M%S}.log".format(datetime.now())
        log_fname = os.path.join(seed_dirname, log_fname)
        logger.debug("logging output to {}".format(log_fname))
        with open(log_fname, 'w') as logf:
            p = subprocess.Popen(cmd_list, stdout=logf, stderr=logf)
            experiment_processes.append(p)

        # run another process including title text features
        cmd = "python {} {} --seed 999 --review-id {} --dataset-seed {} --year {} --titles-cossim --save-best --debug".format(EXPERIMENTS_SCRIPT, seed_dirname, args.paperid, random_seed, args.year)
        logger.debug("running command (in the background): {}...".format(cmd))
        cmd_list = shlex.split(cmd)
        log_fname = "experiments_featuresAvgDistanceAndEFAndAvgCosSim_{:%Y%m%d%H%M%S}.log".format(datetime.now())
        log_fname = os.path.join(seed_dirname, log_fname)
        logger.debug("logging output to {}".format(log_fname))
        with open(log_fname, 'w') as logf:
            p = subprocess.Popen(cmd_list, stdout=logf, stderr=logf)
            experiment_processes.append(p)
        if random_seed < args.max_seed-1:
            logger.debug("experiments are running in the background. this script is no longer concerned with them. moving on to the next random seed\n")
    logger.debug("experiments are running in the background. this script will exit now")
    

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="given one review paper and a range of random seeds, collect all of the candidate papers (haystack) then run experiments")
    parser.add_argument("paperid", help="MAG paper id for the review article that contains the references")
    parser.add_argument("--year", type=int, help="the publication year of the review article")
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
