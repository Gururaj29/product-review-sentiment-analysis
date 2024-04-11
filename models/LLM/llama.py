from stormtrooper import ZeroShotClassifier as classifier

from utils import constants
from utils import utils

class ZeroShotClassifier:
  def __init__(self, model="meta-llama/Llama-2-7b-hf", device="cpu"):
    self.model = classifier(model, device=device).fit(None, [constants.LabelPositive, constants.LabelNegative])
  def fit(self, train_df):
    return
  def predict(self, data_df):
    predictions = self.model.predict(data_df[constants.ColumnData])
    label_classes = utils.get_labels()
    return [label_classes[pred] for pred in predictions]