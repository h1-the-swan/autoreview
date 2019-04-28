# -*- coding: utf-8 -*-
import os
import numpy as np

def load_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state

def prepare_directory(outdir):
    """Make sure the output directory does not exist, then create it.
    """
    outdir = os.path.abspath(outdir)
    if os.path.exists(outdir):
        raise RuntimeError("output path {} already exists!".format(outdir))
    os.mkdir(outdir)

def load_spark_dataframe(path_to_data, spark, fmt=None):
    b = os.path.basename(path_to_data)
    if (fmt and fmt.lower()=='tsv') or 'csv' in b.lower() or 'tsv' in b.lower():
        # ASSUME TAB SEPARATED
        return spark.read.csv(path_to_data, sep='\t')
    else:
        # Assume parquet format
        return spark.read.parquet(path_to_data)

