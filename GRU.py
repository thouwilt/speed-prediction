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

'''
plt.plot(series[:, 0])
plt.title("Speed")
plt.xlabel("Time, s")
plt.ylabel("Speed, km/h")
plt.show()'''

time = np.arange(960, dtype="float32")

split_time = 220
time_train = time[split_time:]
x_train = series[split_time:]
time_valid = time[:split_time]
x_valid = series[:split_time]

window_size = 20
batch_size = 32
shuffle_buffer_size = 740
'''
plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_train, x_train, label='Train Data')
plt.plot(time_valid, x_valid, label='Valid Data')
plt.xlabel("Time, s")
plt.ylabel("Related Speed, m/s")
plt.legend()
plt.grid(True)
plt.show() '''

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


'''
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

# window_size = 30
train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                           input_shape=[None]),
    # tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 200)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1e-3, 0, 1])
plt.show()
'''

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
es = keras.callbacks.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=40, verbose=1, mode='min', baseline=None, restore_best_weights=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape=[None]),
    tf.keras.layers.GRU(32, activation='hard_sigmoid'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

model.compile(loss=tf.keras.losses.mean_squared_error,# Huber(),# "mse",
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999), # SGD(lr=1e-5, momentum=0.9),
              metrics=["mae"])
history = model.fit(dataset, epochs=2000, verbose=1, callbacks=[es])

forecast = []
results = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[:split_time]#  - window_size:]
results = np.array(forecast)[:, 0, 0]
'''
error = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()

plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid, x_valid, label='Input')
plt.plot(time_valid, results, label='Forecasted')'''

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()

plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid, x_valid, label='Input')
plt.plot(time_valid_1, results_1, label='Forecasted')
plt.title("GRU, MAE = " + str(error))
plt.xlabel("Time, s")
plt.ylabel("Speed, m/s")
plt.legend()
plt.grid(True)
plt.show()
