from utils import constants
import pandas as pd
import os
from utils import utils

def load_raw_data(dataset, root_path=None):
    if dataset == constants.DatasetAmazon:
        return load_amazon_raw_data(root_path=root_path)
    if dataset == constants.DatasetYelp:
        return load_yelp_raw_data(root_path=root_path)
    return None, None

def load_amazon_raw_data(root_path=None):
    train_data_path, test_data_path = utils.get_data_filepath(constants.DatasetAmazon, train=True, root_path=root_path), utils.get_data_filepath(constants.DatasetAmazon, train=False, root_path=root_path)
    return load_amazon_raw_data_from_path(train_data_path), load_amazon_raw_data_from_path(test_data_path)

def load_yelp_raw_data(root_path=None):
    train_data_path, test_data_path = utils.get_data_filepath(constants.DatasetYelp, train=True, root_path=root_path), utils.get_data_filepath(constants.DatasetYelp, train=False, root_path=root_path)
    return load_yelp_raw_data_from_path(train_data_path), load_yelp_raw_data_from_path(test_data_path)

def load_amazon_raw_data_from_path(file_path):
    data_df = pd.read_csv(file_path, names=["Rating", "Title", "Review"])
    data_df = pd.concat([data_df["Rating"], data_df["Title"] + "" + data_df["Review"]], axis=1).rename(columns={"Rating": constants.ColumnLabel, 0: constants.ColumnData})
    return data_df.dropna()

def load_yelp_raw_data_from_path(file_path):
    data_df = pd.read_csv(file_path, dtype={"review_text": str, "class_index": int}).rename(columns={'class_index': constants.ColumnLabel, 'review_text': constants.ColumnData})
    return data_df.dropna()

def load_feature(dataset, feature, max_rows=None, binary_labels=False, root_path=None):
    if max_rows is not None:
        train_df, test_df = pd.read_csv(utils.get_data_filepath(dataset, feature=feature, max_rows=max_rows, train=True, root_path=root_path))[:max_rows], pd.read_csv(utils.get_data_filepath(dataset, feature=feature, max_rows=max_rows, train=False, root_path=root_path))[:max_rows]
    else:
        train_df, test_df = pd.read_csv(utils.get_data_filepath(dataset, feature=feature, train=True, root_path=root_path)), pd.read_csv(utils.get_data_filepath(dataset, feature=feature, max_rows=max_rows, train=False, root_path=root_path))
    if binary_labels and dataset == constants.DatasetAmazon:
        # convert 1, 2, 3 ratings as negative and 4 and 5 as positive
        labels = utils.get_labels()
        positive_label, negative_label = labels[constants.LabelPositive], labels[constants.LabelNegative]
        label_substitutions = {1: negative_label, 2: negative_label, 3: positive_label, 4: positive_label, 5: positive_label}
        train_df.replace(label_substitutions, inplace=True)
        test_df.replace(label_substitutions, inplace=True)
    return train_df, test_df

def load_all_features(dataset, max_rows=None, binary_labels=False):
    feature_dfs = {}
    for feature in utils.get_all_features():
        feature_dfs[feature] = load_feature(dataset, feature, max_rows=max_rows, binary_labels=binary_labels)
    return feature_dfs
    