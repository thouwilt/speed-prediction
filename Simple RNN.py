import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_excel('log1_ds.xlsx')
data.head(0)
data.info()

column = data.iloc[:, [0]].values
series = column.T
series = series.ravel()

time = np.arange(960, dtype="float32")
'''
split_time = 860
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 860
'''

split_time = 220
time_train = time[split_time:]
x_train = series[split_time:]
time_valid = time[:split_time]
x_valid = series[:split_time]

window_size = 20
batch_size = 32
shuffle_buffer_size = 740


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
# определить скорость обучения
'''
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                           input_shape=[None]),
    tf.keras.layers.SimpleRNN(20, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
plt.show()
# we can see that the optimal learning rate is 10^(-5)
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
'''
tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size=32, shuffle_buffer=shuffle_buffer_size)  # '128'
es = keras.callbacks.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=40, verbose=1, mode='min',
                                             baseline=None, restore_best_weights=False)
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                           input_shape=[None]),
    tf.keras.layers.SimpleRNN(30, activation='hard_sigmoid'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=10000, callbacks=[es])

forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[:split_time]  # -window_size:]
# forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]
'''
error = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()

plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid, x_valid, label='Input')
plt.plot(time_valid, results, label='Forecasted')
plt.title("RNN, MAE = " + str(error))
plt.xlabel("Time, s")
plt.ylabel("Speed, m/s")
plt.legend()
plt.grid(True)
plt.show()
'''

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]
print(results_1)
error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()

plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid, x_valid, label='Input')
plt.plot(time_valid_1, results_1, label='Forecasted')
plt.title("RNN, MAE = " + str(error))
plt.xlabel("Time, s")
plt.ylabel("Speed, m/s")
plt.legend()
plt.grid(True)
plt.show()

error1 = tf.keras.metrics.mean_squared_error(x_valid, results).numpy()
print(error1)
