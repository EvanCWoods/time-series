import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import keras
from keras import layers
from keras.layers import Dense, LSTM, Conv1D
from keras.optimizers import Adam
from keras.activations import relu
from keras.losses import mae

time = []
sunspots = []

with open('Sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time.append(int(row[0]))

time = np.array(time)
sunspots = np.array(sunspots)
print(len(time))

split_point = 3000
train_data = sunspots[:split_point]
train_time = time[:split_point]
test_data = sunspots[split_point:]
test_time = time[split_point:]

window_size = 64
batch_size = 256
shuffle_window_buffer = 1000


def windowed_data(series, window, batch, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch).prefetch(1)


train_data = windowed_data(series=train_data,
                           window=window_size,
                           batch=batch_size,
                           shuffle_buffer=shuffle_window_buffer)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.97:
            print('97% accuracy reached, stopping training.')
            self.model.stop_training = True


callback = MyCallback()

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=5,
                           strides=1,
                           padding='causal',
                           input_shape=[None, 1],
                           activation=relu),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(64, activation=relu),
    tf.keras.layers.Dense(32, activation=relu),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data,
                    epochs=1,
                    callbacks=[callback])


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

series=test_data

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_point - window_size:-1, -1, 0]
tf.keras.metrics.mean_absolute_error(test_data, rnn_forecast).numpy()