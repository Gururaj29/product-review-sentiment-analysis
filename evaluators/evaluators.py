import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from utils import constants

get_accuracy = lambda df, classifier: accuracy_score(df[constants.ColumnLabel], classifier.predict(df.drop([constants.ColumnLabel], axis=1)))

# Method for kfold parameter tuning and plotting accuracies - prints and plots training and validation accuracies for any model and params
def kfold_parameter_tune(full_train_df, k=5, parameters = [], parameter_name = "", model_init = None, ax = None, model_name = "", title = "", print_accuracies = True, plot_accuracies = True):
    train_acc_all = []
    val_acc_all = []

    # df columns
    avg_train_acc_col = "Training accuracies"
    std_train_acc_col = "Std of training accuracies"
    avg_val_acc_col = "Val accuracies"
    std_val_acc_col = "Std of val accuracies"

    kf = KFold(n_splits = k)
    accuracies = {parameter_name: [], avg_train_acc_col: [], std_train_acc_col: [], avg_val_acc_col: [], std_val_acc_col:[]}
    for param in tqdm(parameters):
        train_acc, val_acc = [], []
        for train_index, val_index in kf.split(full_train_df):
            full_train_df, model = model_init(full_train_df, param)
            train_df, val_df = full_train_df.iloc[train_index], full_train_df.iloc[val_index]
            model.fit(train_df)
            train_acc.append(get_accuracy(train_df, model))
            val_acc.append(get_accuracy(val_df, model))
        avg_train_acc, std_train_acc = np.average(train_acc), np.std(train_acc)
        avg_val_acc, std_val_acc = np.average(val_acc), np.std(val_acc)
        accuracies[avg_train_acc_col].append(avg_train_acc)
        accuracies[std_train_acc_col].append(std_train_acc)
        accuracies[avg_val_acc_col].append(avg_val_acc)
        accuracies[std_val_acc_col].append(std_val_acc)
        accuracies[parameter_name].append(param)

    accuracies_df = pd.DataFrame(accuracies, index=parameters)

    if print_accuracies:
      print("\n"* 2)
      print(tabulate(accuracies_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    if plot_accuracies:
      if not ax:
        fig, ax = plt.subplots()
      train_line, = ax.plot(accuracies[parameter_name], accuracies[avg_train_acc_col], marker='.', label=("%s_"%model_name if model_name else "") + "TrainingAccuracy")
      val_line, = ax.plot(accuracies[parameter_name], accuracies[avg_val_acc_col], marker='.', label=("%s_"%model_name if model_name else "") + "ValidationAccuracy")
      ax.set_xlabel(parameter_name)
      ax.set_ylabel('Accuracy')
      plt.title(title)
      plt.xticks(rotation=35)
      ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1))

    return accuracies_df, ax