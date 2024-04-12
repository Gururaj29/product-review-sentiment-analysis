from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer

from utils import constants
from utils import utils

BERT_MODEL = ""
RoBERTa_MODEL = ""
DISTILBERT_MODEL = "distilbert/distilbert-base-uncased"

class ZeroShotClassifier:
  def __init__(self, model="", device=-1):
    self.classifier = pipeline(DISTILBERT_MODEL, model=model, device=device, max_length=512, truncation=True)
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
       tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL, use_fast=True)
       labels = utils.get_labels()
       self.class_to_label = {
         "LABEL_0": labels[constants.LabelNegative],
         "LABEL_1": labels[constants.LabelPositive]
       }
       self.classifier = pipeline("text-classification", model=model_path, tokenizer=tokenizer, device=device, max_length=512, truncation=True)
    
    def fit(self, train_df):
        # it's already trained
        return
    
    def predict(self, data_df):
        predictions = []
        for review in tqdm(data_df[constants.ColumnData]):
            prediction = self.classifier(review)[0]
            predicted_class = prediction["label"]
            predictions.append(self.class_to_label[predicted_class])
        return predictions