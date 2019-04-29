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
    if (fmt and fmt.lower()=='tsv') or ('parquet' not in path_to_data.lower() and ('csv' in path_to_data.lower() or 'tsv' in path_to_data.lower())):
        # ASSUME TAB SEPARATED
        return spark.read.csv(path_to_data, sep='\t', header=True)
    else:
        # Assume parquet format
        return spark.read.parquet(path_to_data)

def get_year(x):
    if x:
        return x.year
    else:
        return None

def save_pandas_dataframe_to_pickle(df, outfname):
    if 'year' not in df.columns:
        df['year'] = df['pub_date'].apply(get_year)
    columns_rename = {
        'ID': 'Paper_ID',
        'flow': 'EF'
    }
    df.rename(columns=columns_rename, inplace=True)
    df.to_pickle(outfname)
