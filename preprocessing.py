from preprocessors import loaders
from preprocessors import features
from utils import constants
from utils import utils

dataset = constants.DatasetAmazon
MAX_ROWS = 1000

train_data_df, test_data_df = loaders.load_raw_data(dataset)

print("Shapes of the train and test datasets are: ", train_data_df.shape, test_data_df.shape)

for feature in utils.get_all_features():
    print("Processing feature:", feature, "...", MAX_ROWS)
    train_df, test_df = features.extract_feature(feature, train_data_df, test_data_df, max_rows=MAX_ROWS)
    train_df.to_csv(utils.get_data_filepath(dataset, train=True, feature=feature))
    test_df.to_csv(utils.get_data_filepath(dataset, train=False, feature=feature))


dataset = constants.DatasetYelp
MAX_ROWS = 1000

train_data_df, test_data_df = loaders.load_raw_data(dataset)

print("Shapes of the train and test datasets are: ", train_data_df.shape, test_data_df.shape)

for feature in utils.get_all_features():
    print("Processing feature:", feature, "...", MAX_ROWS)
    train_df, test_df = features.extract_feature(feature, train_data_df, test_data_df, max_rows=MAX_ROWS)
    train_df.to_csv(utils.get_data_filepath(dataset, train=True, feature=feature))
    test_df.to_csv(utils.get_data_filepath(dataset, train=False, feature=feature))