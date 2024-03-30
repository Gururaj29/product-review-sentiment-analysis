from utils import constants
import pandas as pd
import os
from utils import utils

def load_raw_data(dataset):
    if dataset == constants.DatasetAmazon:
        return load_amazon_raw_data()
    if dataset == constants.DatasetYelp:
        return load_yelp_raw_data()
    return None, None

def load_amazon_raw_data():
    train_data_path, test_data_path = utils.get_data_filepath(constants.DatasetAmazon, train=True), utils.get_data_filepath(constants.DatasetAmazon, train=False)
    return load_amazon_raw_data_from_path(train_data_path), load_amazon_raw_data_from_path(test_data_path)

def load_yelp_raw_data():
    train_data_path, test_data_path = utils.get_data_filepath(constants.DatasetYelp, train=True), utils.get_data_filepath(constants.DatasetYelp, train=False)
    return load_yelp_raw_data_from_path(train_data_path), load_yelp_raw_data_from_path(test_data_path)

def load_amazon_raw_data_from_path(file_path):
    data_df = pd.read_csv(file_path, names=["Rating", "Title", "Review"])
    data_df = pd.concat([data_df["Rating"], data_df["Title"] + "" + data_df["Review"]], axis=1).rename(columns={"Rating": constants.ColumnLabel, 0: constants.ColumnData})
    return data_df.dropna()

def load_yelp_raw_data_from_path(file_path):
    data_df = pd.read_csv(file_path, dtype={"review_text": str, "class_index": int}).rename(columns={'class_index': constants.ColumnLabel, 'review_text': constants.ColumnData})
    return data_df.dropna()

def load_feature(dataset, feature, max_rows=None, binary_labels=False):
    if max_rows is not None:
        train_df, test_df = pd.read_csv(utils.get_data_filepath(dataset, feature=feature, max_rows=max_rows, train=True))[:max_rows], pd.read_csv(utils.get_data_filepath(dataset, feature=feature, max_rows=max_rows, train=False))[:max_rows]
    else:
        train_df, test_df = pd.read_csv(utils.get_data_filepath(dataset, feature=feature, train=True)), pd.read_csv(utils.get_data_filepath(dataset, feature=feature, max_rows=max_rows, train=False))
    if binary_labels and dataset == constants.DatasetAmazon:
        # convert 1, 2, 3 ratings as negative and 4 and 5 as positive
        train_df.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 2}, inplace=True)
        test_df.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 2}, inplace=True)
    return train_df, test_df

def load_all_features(dataset, max_rows=None, binary_labels=False):
    feature_dfs = {}
    for feature in utils.get_all_features():
        feature_dfs[feature] = load_feature(dataset, feature, max_rows=max_rows, binary_labels=binary_labels)
    return feature_dfs
    