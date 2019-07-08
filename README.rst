Automated Lit Review
========================

Jason Portenoy
2018

Library structure is based on https://github.com/kennethreitz/samplemod.

Starting with a list of seed papers, get candidate papers by following in- and out-citations (2 degrees).
Then, train a classifier to rank the candidate papers.
Repeat this a number of times to get an aggregate ranking for many candidate papers.

Example script in `scripts/run_autoreview.py`

Inputs:
- List of paper IDs for the seed set.
- Data for paper citations.
- Paper data to be used as features for the classifiers (e.g., clusters, eigenfactor, titles, etc.)

Parameters:
- Size of the initial split
- Number of times to perform the overall process of collecting candidate papers and training a classifier

Output:
- List of papers not in the seed set, ordered descending by relevance score.
