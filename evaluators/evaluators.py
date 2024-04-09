import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import time
import os

from utils import utils
from utils import constants
from visualization import visualization

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

def evaluate(model, train_df, test_df, evaluator_name='', print_time=True):
    start_time = time.process_time()
    model.fit(train_df)
    train_time = time.process_time() - start_time
    if print_time:
        print("Time taken (in seconds) for training: ", train_time)
    return EvaluatorResults(model, evaluator_name = evaluator_name, train_df=train_df, test_df=test_df, train_time=train_time, print_time=print_time)

class EvaluatorResults:
    def __pred(self, model, df, true_labels, print_time, dataset):
        start_time = time.process_time()
        pred_labels = model.predict(df.drop([constants.ColumnLabel], axis=1))
        pred_time = time.process_time() - start_time
        
        if print_time:
           print("Time taken (in seconds) for predicting %s dataset: %f"% (dataset, pred_time))

        labels = utils.get_labels()
        cr = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0.0)
        cm = confusion_matrix(true_labels, pred_labels, labels=[labels[constants.LabelPositive], labels[constants.LabelNegative]])
        return pred_labels, pred_time, cr, cm
        
    def __init__(self, model, evaluator_name = '', train_df = None, test_df = None, 
                train_time = None, print_time = True, train_output_filepath=None, test_output_filepath=None):
        self.train_time = train_time
        self.name = evaluator_name

        # Train attributes
        self.train_true_labels = None
        self.train_pred_labels = None
        self.train_pred_time = None
        self.train_accuracy = None
        self.train_classification_report = None
        self.train_confusion_matrix = None

        # Test attributes
        self.test_true_labels = None
        self.test_pred_labels = None
        self.test_pred_time = None
        self.test_accuracy = None
        self.test_classification_report = None
        self.test_confusion_matrix = None

        if train_df is not None:
           self.train_true_labels = train_df[constants.ColumnLabel]
           self.train_pred_labels, self.train_pred_time, self.train_classification_report, self.train_confusion_matrix = self.__pred(
              model, train_df, self.train_true_labels, print_time, "Train")
           self.train_accuracy = self.train_classification_report['accuracy']

        if test_df is not None:
           self.test_true_labels = test_df[constants.ColumnLabel]
           self.test_pred_labels, self.test_pred_time, self.test_classification_report, self.test_confusion_matrix = self.__pred(
              model, test_df, self.test_true_labels, print_time, "Test")
           self.test_accuracy = self.test_classification_report['accuracy']
        
        if train_output_filepath is not None:
           if self.train_true_labels is not None and self.train_pred_labels is not None:
              pd.DataFrame({"True labels": self.train_true_labels, "Predicted labels": self.train_pred_labels}).to_csv(train_output_filepath)

        if test_output_filepath is not None:
           if self.test_true_labels is not None and self.test_pred_labels is not None:
              pd.DataFrame({"True labels": self.test_true_labels, "Predicted labels": self.test_pred_labels}).to_csv(test_output_filepath)
           

    def plot_confusion_matrix(self, isTrainDf):
       cm = self.train_confusion_matrix if isTrainDf else self.test_confusion_matrix
       visualization.plot_confusion_matrices(cm)
       return
    
    def get_classification_report(self, isTrainDf):
       return self.train_classification_report if isTrainDf else self.test_classification_report
    
    def get_accuracy(self, isTrainDf):
       return self.train_accuracy if isTrainDf else self.test_accuracy
    
    def get_train_time(self):
       return self.train_time
    
    def get_prediction_time(self, isTrain):
       return self.train_pred_time if isTrain else self.test_pred_time
    
    def get_predictions(self, isTrain):
       return self.train_pred_labels if isTrain else self.test_pred_labels

    def get_accuracies(self):
       return self.train_accuracy, self.test_accuracy
    
    def get_evaluator_name(self):
       return self.name
    
    def display_classification_report(self, isTrain, include_avg=True):
        cr = self.train_classification_report if isTrain else self.test_classification_report

        labels = utils.get_labels()
        labels_to_display_name = {str(v): k for k, v in labels.items()}
        report = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
        index = []

        def add_row(report, row_dict):
            for key in row_dict:
                report[key].append(row_dict[key])

        for row_key in cr:
            if row_key != 'accuracy':
                if row_key in labels_to_display_name:
                    # classes
                    index.append(labels_to_display_name[row_key])
                    add_row(report, cr[row_key])
                elif include_avg:
                    # averages
                    index.append(row_key)
                    add_row(report, cr[row_key])

        df = pd.DataFrame(report, index=index)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
        return

    def plot_confusion_matrices(self):
        visualization.plot_confusion_matrices(self.train_confusion_matrix, self.test_confusion_matrix)
        return

def batch_evaluate(model, model_name, train_df, test_df, batch_size, root_path, dataset, run_id):
    batch_no = 1
    dir_path = utils.get_predictions_output_path(root_path=root_path, model_name=model_name, dataset=dataset, run_id=run_id)
    total_batches = len(train_df) // batch_size if len(train_df)%batch_size == 0 else len(train_df)//batch_size + 1
    for i in range(0, len(train_df), batch_size):
        batch_train_df, batch_test_df = train_df[i: min(i+batch_size, len(train_df))], None if i >= len(test_df) else test_df[i: min(i+batch_size, len(test_df))]
        try:
            EvaluatorResults(model=model, train_df=batch_train_df, test_df=batch_test_df, train_time=1, train_output_filepath=os.path.join(dir_path, "train_predictions_%d.csv"%batch_no), test_output_filepath=os.path.join(dir_path, "test_predictions_%d.csv"%batch_no))
        except Exception as e:
           print("Exception!", e)
        print("Completed batch#: %d, remaining: %d" %(batch_no, total_batches - batch_no))
        batch_no += 1
      