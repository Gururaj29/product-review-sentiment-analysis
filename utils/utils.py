import os
from utils import constants
import pathlib
import numpy as np
import pandas as pd

def get_data_filepath(dataset, train=True, feature=None, max_rows=None):
    filename = "train.csv" if train else "test.csv"
    dirpath = os.path.join(constants.RAW_DATA_PATH, dataset) if feature is None else (
        os.path.join(constants.FEATURES_PATH, dataset, feature) if max_rows is None or max_rows <= constants.SmallFeatureLimit else os.path.join(constants.ALL_FEATURES_PATH, dataset, feature))
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
    return os.path.join(dirpath, filename)

def get_all_features():
    return {constants.FeatureCountVectorizer, constants.FeatureGloVe, constants.FeatureTFIDF}

def split_train_data(train_df, split_percent=0.8):
    return np.split(train_df.sample(frac=1, random_state=42), [int(split_percent*len(train_df))])