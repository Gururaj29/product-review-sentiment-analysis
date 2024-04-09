from stormtrooper import ZeroShotClassifier

from utils import constants
from utils import utils

class BARTZeroShotClassifier:
  def __init__(self, model="facebook/bart-large-mnli", device="cpu"):
    self.model = ZeroShotClassifier(model, device=device).fit(None, [constants.LabelPositive, constants.LabelNegative])
  def fit(self, train_df):
    return
  def predict(self, data_df):
    predictions = self.model.predict(data_df[constants.ColumnData])
    label_classes = utils.get_labels()
    return [label_classes[pred] for pred in predictions]