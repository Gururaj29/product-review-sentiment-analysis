from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from utils import constants
import numpy as np

class Classifier:
  def __init__(self, labels, output_activation = 'softmax', optimizer='adam', learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    self.label_encoder = LabelEncoder()
    self.label_encoder.fit(labels)
    self.model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(len(self.label_encoder.classes_), tf.keras.activations.softmax)])
    self.model.compile(optimizer=self.__get_optimizer(optimizer, learning_rate), loss=loss, metrics=metrics)

  def __get_optimizer(self, optimizer, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

  def fit(self, train_df):
    X = train_df.drop([constants.ColumnLabel], axis=1).values
    Y = self.label_encoder.transform(train_df[constants.ColumnLabel])
    self.model.fit(X, Y, epochs=10)

  def predict(self, test_df):
    if constants.ColumnLabel in test_df.columns:
      test_df = test_df.drop([constants.ColumnLabel], axis=1)
    y_pred = self.model.predict(test_df.values)
    y_pred = np.argmax(y_pred, axis=1)
    return self.label_encoder.inverse_transform(y_pred)