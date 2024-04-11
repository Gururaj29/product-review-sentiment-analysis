from tqdm import tqdm
from transformers import pipeline

from utils import constants
from utils import utils

class ZeroShotClassifier:
  def __init__(self, model="openai-community/gpt2-large", device=-1):
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