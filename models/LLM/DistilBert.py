from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer

from utils import constants
from utils import utils

__pretrained_llm = "distilbert/distilbert-base-uncased"

class ZeroShotClassifier:
  def __init__(self, model=__pretrained_llm, device=-1):
    self.classifier = pipeline("zero-shot-classification", model=model, device=device)
  def fit(self, train_df):
    return
  def predict(self, data_df):
    label_classes = utils.get_labels()
    labels = list(label_classes.keys())
    predictions = []

    for review in tqdm(data_df[constants.ColumnData]):
      prediction = self.classifier(review, candidate_labels=labels)
      label, scores = prediction["labels"], prediction["scores"]
      predicted_class = label[0] if scores[0] >= scores[1] else label[0]
      predictions.append(label_classes[predicted_class])
    
    return predictions

class FineTunedClassifier:
    def __init__(self, model_path, device=-1):
       tokenizer = AutoTokenizer.from_pretrained(__pretrained_llm, use_fast=True)
       labels = utils.get_labels()
       self.class_to_label = {
         "LABEL_0": labels[constants.LabelNegative],
         "LABEL_1": labels[constants.LabelPositive]
       }
       self.classifier = pipeline("text-classification", model=model_path, tokenizer=tokenizer)
    
    def fit():
        # it's already trained
        return
    
    def predict(self, data_df):
        predictions = []
        for review in tqdm(data_df[constants.ColumnData]):
            prediction = self.classifier(review)[0]
            predicted_class = prediction["label"]
            predictions.append(self.class_to_label[predicted_class])
        return predictions