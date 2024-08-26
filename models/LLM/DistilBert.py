from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer

from utils import constants
from utils import utils

BERT_MODEL = "google-bert/bert-base-uncased"
RoBERTa_MODEL = "FacebookAI/roberta-base"
DISTILBERT_MODEL = "distilbert/distilbert-base-uncased"

class ZeroShotClassifier:
  def __init__(self, model=DISTILBERT_MODEL, device=-1, max_length=512, truncation=True):
    self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=AutoTokenizer.from_pretrained(model, use_fast=False), device=device, max_length=max_length, truncation=True, padding=True)
    self.max_length = max_length
    self.truncation = truncation
  def fit(self, train_df):
    return
  def predict(self, data_df):
    label_classes = utils.get_labels()
    labels = list(label_classes.keys())
    predictions = []

    for review in tqdm(data_df[constants.ColumnData]):
      sentence = review
      if self.truncation:
        sentence = review[:self.max_length]         
      prediction = self.classifier(sentence, candidate_labels=labels, max_length=self.max_length, padding=True, truncation=True)
      label, scores = prediction["labels"], prediction["scores"]
      predicted_class = label[0] if scores[0] >= scores[1] else label[0]
      predictions.append(label_classes[predicted_class])
    
    return predictions

class FineTunedClassifier:
    def __init__(self, model_path, device=-1, max_length = 512, truncation = True):
       labels = utils.get_labels()
       self.class_to_label = {
         "LABEL_0": labels[constants.LabelNegative],
         "LABEL_1": labels[constants.LabelPositive]
       }
       self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device, max_length=max_length, truncation=True)
       self.max_length = max_length
       self.truncation = truncation
    
    def fit(self, train_df):
        # it's already trained
        return
    
    def predict(self, data_df):
        predictions = []
        for review in tqdm(data_df[constants.ColumnData]):
            sentence = review
            if self.truncation:
                sentence = review[:self.max_length]         
            prediction = self.classifier(sentence, max_length=self.max_length, padding=True, truncation=True)[0]
            predicted_class = prediction["label"]
            predictions.append(self.class_to_label[predicted_class])
        return predictions