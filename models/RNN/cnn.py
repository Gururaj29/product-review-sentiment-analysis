from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from utils import constants
import numpy as np

class Classifier:
    def __init__(self, labels, output_activation='softmax', optimizer='adam', learning_rate=0.001, loss='sparse_categorical_crossentropy', metrics=['accuracy'], input_shape=8628):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.model = self.__create_cnn_model(len(self.label_encoder.classes_), input_shape)
        self.model.compile(optimizer=self.__get_optimizer(optimizer, learning_rate), loss=loss, metrics=metrics)

    def __create_cnn_model(self, num_classes, input_shape):
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #     tf.keras.layers.MaxPooling2D((2, 2)),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D((2, 2)),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(num_classes, activation='softmax')
        # ])
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def __get_optimizer(self, optimizer, learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def fit(self, train_data):
        X = train_data.drop(columns=['Label'], axis=1).values
        Y = self.label_encoder.transform(train_data[constants.ColumnLabel])
        # X = X.reshape(X.shape[0], 28, 28, 1)  # Reshape for Conv2D input
        self.model.fit(X, Y, epochs=10)

    def predict(self, test_data):
        if constants.ColumnLabel in test_data.columns:
            test_data = test_data.drop(columns=['Label'], axis=1).values
        # X_test = test_data.values.reshape(test_data.shape[0], 28, 28, 1)  # Reshape for Conv2D input
        y_pred = self.model.predict(test_data)
        y_pred = np.argmax(y_pred, axis=1)
        return self.label_encoder.inverse_transform(y_pred)