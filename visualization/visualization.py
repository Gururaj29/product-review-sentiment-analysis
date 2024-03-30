import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from utils import constants
import pandas as pd

def plot_label_distribution(train_df, test_df, binary_labels=True):
  train_labels = dict(train_df[constants.ColumnLabel].value_counts())
  test_labels = dict(test_df[constants.ColumnLabel].value_counts())
  train_label_keys = list(train_labels.keys())
  train_label_count = list(train_labels.values())

  test_label_keys = list(test_labels.keys())
  test_label_count = list(test_labels.values())

  if binary_labels:
    train_label_keys = ['Positive' if i>1 else 'Negative' for i in train_label_keys]
    test_label_keys = ['Positive' if i>1 else 'Negative' for i in test_label_keys]

  # Plotting the pie charts
  plt.figure(figsize=(12, 5))

  # First subplot
  plt.subplot(1, 2, 1)
  plt.pie(train_label_count, labels=train_label_keys, autopct='%1.1f%%', startangle=140)
  plt.title('Train labels distribution')

  # Second subplot
  plt.subplot(1, 2, 2)
  plt.pie(test_label_count, labels=test_label_keys, autopct='%1.1f%%', startangle=140)
  plt.title('Test labels distribution')

  plt.axis('equal')
  plt.show()

# Plot bar figure of training and validation/testing accuracies
def plot_accuracies_bar_figure(accuracies_df, parameter_name):
  at = list(accuracies_df.columns)

  accuracies = {
      "AccuracyType": [],
      parameter_name: [],
      "Accuracy": [],
  }

  for i in accuracies_df.index:
    accuracies["AccuracyType"] += at
    accuracies[parameter_name] += [i, i]
    accuracies["Accuracy"] += list(accuracies_df.loc[i].values)

  accuracies_df_for_plt = pd.DataFrame(accuracies)

  fig, ax = plt.subplots()
  sns.barplot(x=parameter_name, hue="AccuracyType", y="Accuracy", data=accuracies_df_for_plt)
  plt.title("Bar figure for accuracies")
  plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.25))
  plt.xticks(rotation=25)
  plt.show()