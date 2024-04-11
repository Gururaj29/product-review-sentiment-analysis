import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np

from utils import constants

class Classifier:
    def __init__(self, labels, output_activation='softmax', optimizer='adam', learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(len(self.label_encoder.classes_), activation=output_activation)])
        self.model.compile(optimizer=self.__get_optimizer(optimizer, learning_rate), loss=loss, metrics=metrics)

    def __get_optimizer(self, optimizer, learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def fit(self, train_df, time_steps=1, epochs=10):
        X = train_df.drop([constants.ColumnLabel], axis=1).values
        Y = self.label_encoder.transform(train_df[constants.ColumnLabel])

        # Reshape input data to be 3-dimensional (batch_size, time_steps, input_dim)
        X = X.reshape(X.shape[0], time_steps, -1)

        self.model.fit(X, Y, epochs=epochs)

    def predict(self, test_df, time_steps=1):
        if constants.ColumnLabel in test_df.columns:
            test_df = test_df.drop([constants.ColumnLabel], axis=1)

        X = test_df.values.astype(np.float32)  # Convert data to float32
        # Reshape input data to be 3-dimensional (batch_size, time_steps, input_dim)
        X = X.reshape(X.shape[0], time_steps, -1)

        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return self.label_encoder.inverse_transform(y_pred)