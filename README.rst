########################
Automated Lit Review
########################

Jason Portenoy
2018

This is code and sample data accompanying the paper:

`Supervised Learning for Automated Literature Review <http://ceur-ws.org/Vol-2414/paper8.pdf>`_

published in the proceedings of the 4th Joint Workshop on Bibliometric-enhanced Information Retrieval and Natural Language Processing for Digital Libraries (BIRNDL 2019)
co-located with the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2019) 

Starting with a list of seed papers, get candidate papers by following in- and out-citations (2 degrees).
Then, train a classifier to rank the candidate papers.
Repeat this a number of times to get an aggregate ranking for many candidate papers.

Example script in ``scripts/run_autoreview.py``

Inputs:

- List of paper IDs for the seed set.
- Data for paper citations.
- Paper data to be used as features for the classifiers (e.g., clusters, eigenfactor, titles, etc.)

Parameters:

- Size of the initial split
- Number of times to perform the overall process of collecting candidate papers and training a classifier

Output:

- List of papers not in the seed set, ordered descending by relevance score.

Installation
============

Install via PyPI::

        pip install autoreview

Example
=======

- Apache Spark (https://spark.apache.org/downloads.html) must be installed to run the example.

- The environment variable ``SPARK_HOME`` must be set (preferably in a ``.env`` file) with the path to Spark.

  + Java version 8 is required to be used with Spark. Make sure Java 8 is installed and point to its path with the environment variable ``JAVA_HOME``.

  + Example ``.env`` file::

        SPARK_HOME=/home/spark-2.4.0-bin-hadoop2.7
        JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

- Create a virtual environment and install the required libraries::

        virtualenv venv
        source venv/bin/activate
        pip install -r requirements.txt

- Run the full autoreview pipeline using sample data::

        python scripts/run_autoreview.py --id-list sample_data/sample_IDs_MAG.txt --citations sample_data/MAG_citations_sample --papers sample_data/MAG_papers_sample --sample-size 15 --random-seed 999 --id-colname Paper_ID --cited-colname Paper_Reference_ID --outdir sample_data/sample_output --debug

- This is just meant to show how the system operates. It will not provide meaningful results with such a small sample of paper and citation data.

- It will output the top predictions in ``sample_data/sample_output/predictions.tsv``.

Development
============

For new releases::

        # increment the version number
        bump2version patch

Replace ``patch`` with ``minor`` or ``major`` as needed. Then::

        # push new release to github
        git push --tags

        # build and upload to PyPI
        python setup.py sdist bdist_wheel
        twine check dist/*
        twine upload dist/*

