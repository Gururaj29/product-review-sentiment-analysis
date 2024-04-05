import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from utils import constants
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

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

# Plot bar figure of training and validation/testing accuracies
def plot_pred_time_bar_figure(accuracies_df, parameter_name):
  at = list(accuracies_df.columns)

  accuracies = {
      "TimeTakenType": [],
      parameter_name: [],
      "TimeTaken": [],
  }

  for i in accuracies_df.index:
    accuracies["TimeTakenType"] += at
    accuracies[parameter_name] += [i, i]
    accuracies["TimeTaken"] += list(accuracies_df.loc[i].values)

  accuracies_df_for_plt = pd.DataFrame(accuracies)

  fig, ax = plt.subplots()
  sns.barplot(x=parameter_name, hue="TimeTakenType", y="TimeTaken", data=accuracies_df_for_plt)
  plt.title("Bar figure for prediction time")
  plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.25))
  plt.xticks(rotation=25)
  plt.show()

def plot_bar_figure(df, x_label, y_label, title):
    fig, ax = plt.subplots()
    sns.barplot(x=x_label, hue="Dataset", y=y_label, data=df)
    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.25))
    plt.xticks(rotation=25)
    plt.show()

def plot_accuracies_from_results(results, x_label):
   data = []
   for r in results:
      data.append({x_label: r.get_evaluator_name(), "Dataset": "Train", "Accuracy": r.get_accuracy(True)})
      data.append({x_label: r.get_evaluator_name(), "Dataset": "Test", "Accuracy": r.get_accuracy(False)})
   df = pd.DataFrame(data)
   plot_bar_figure(df, x_label, "Accuracy", "Bar figure for accuracies")

def plot_pred_time_from_results(results, x_label):
    data = []
    for r in results:
        data.append({x_label: r.get_evaluator_name(), "Dataset": "Train", "Prediction Time": r.get_prediction_time(True)})
        data.append({x_label: r.get_evaluator_name(), "Dataset": "Test", "Prediction Time": r.get_prediction_time(False)})
    df = pd.DataFrame(data)
    plot_bar_figure(df, x_label, "Prediction Time", "Bar figure for prediction time")

def plot_confusion_matrices(*confusion_matrices):
    if len(confusion_matrices) == 1:
        ConfusionMatrixDisplay(confusion_matrices[0], display_labels=["Positive", "Negative"]).plot()
        plt.show()
    elif len(confusion_matrices) == 2:
        f, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')

        # plotting train confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrices[0],
                                    display_labels=["Positive", "Negative"])
        disp.plot(ax=axes[0], xticks_rotation=45)
        disp.ax_.set_title("Train dataset")
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')

        # plotting test confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrices[1],
                            display_labels=["Positive", "Negative"])
        disp.plot(ax=axes[1], xticks_rotation=45)
        disp.ax_.set_title("Test dataset")
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')

        f.text(0.5, 0, 'Predicted label', ha='left')
        plt.subplots_adjust(wspace=0.40, hspace=0.1)
        plt.show()
