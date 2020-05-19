import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import keras
import datetime

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
    # tf.keras.layers.GRU(40, return_sequences=True),
    tf.keras.layers.GRU(30),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 30])
plt.show()


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
es = keras.callbacks.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=40, verbose=1, mode='min', baseline=None, restore_best_weights=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape=[None]),
    # tf.keras.layers.GRU(40, return_sequences=True),
    tf.keras.layers.GRU(40, activation='hard_sigmoid'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

model.compile(loss=tf.keras.losses.mean_squared_error,# Huber(),# "mse",
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999), # SGD(lr=1e-5, momentum=0.9),
              metrics=["mae"])
history = model.fit(dataset, epochs=1000, verbose=1, callbacks=[es])
'''
# LSTM
# tf.keras.backend.clear_session()
# tf.random.set_seed(51)
# np.random.seed(51)
#
# dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# es = keras.callbacks.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=40, verbose=1, mode='min', baseline=None, restore_best_weights=False)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),  # butch size, number of timestamps, series dimensionality
#                            input_shape=[None]),
#     # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='hard_sigmoid')),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='hard_sigmoid')),
#     tf.keras.layers.Dense(1),
#     tf.keras.layers.Lambda(lambda x: x * 100.0)
# ])
#
# model.compile(loss=tf.keras.losses.Huber(),
#               optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
#               metrics=["mae"])
# history = model.fit(dataset, epochs=1000, verbose=1, callbacks=[es])


# Simple RNN
# tf.keras.backend.clear_session()
# tf.random.set_seed(51)
# np.random.seed(51)
# dataset = windowed_dataset(x_train, window_size, batch_size=32, shuffle_buffer=shuffle_buffer_size)  # '128'
# es = keras.callbacks.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=40, verbose=1, mode='min',
#                                              baseline=None, restore_best_weights=False)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                            input_shape=[None]),
#     tf.keras.layers.SimpleRNN(30, activation='hard_sigmoid'),
#     tf.keras.layers.Dense(1),
#     tf.keras.layers.Lambda(lambda x: x * 100.0)
# ])
#
# optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
# model.compile(loss=tf.keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
# history = model.fit(dataset, epochs=10000, callbacks=[es])


# MLP
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
es = keras.callbacks.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=40, verbose=1, mode='min', baseline=None, restore_best_weights=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=[window_size]),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=2e-2, momentum=0.9))
model.fit(dataset, epochs=2000, verbose=1, callbacks=[es])




step_predict = 1

forecast = []
results = []
now = datetime.datetime.now()
for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960

print('MAE 1  sec  '+str(error)+':')
print('время при 1 секунде:"  '+str(ti)+':')
plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid, x_valid, label='Input')
plt.plot(time_valid_1, results_1, label='Forecasted')
plt.title("LSTM, MAE = " + str(error))
plt.xlabel("Time, s")
plt.ylabel("Speed, m/s")
plt.legend()
plt.grid(True)
plt.show()

step_predict = 2

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960

print('MAE 2  sec  '+str(error)+':')
print('время при 2 секунде:"  '+str(ti)+':')

# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 3

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960

print('MAE 3  sec  '+str(error)+':')
print('время при 3 секунде:"  '+str(ti)+':')

# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 4

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))
#
# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960

print('MAE 4  sec  '+str(error)+':')
print('время при 4 секунде:"  '+str(ti)+':')

# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 5

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960

print('MAE 5  sec  '+str(error)+':')
print('время при 5 секунде:"  '+str(ti)+':')
# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 6

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960

print('MAE 6  sec  '+str(error)+':')
print('время при 6 секунде:"  '+str(ti)+':')

# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 7

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960
print('MAE 7  sec  '+str(error)+':')
print('время при 7 секунде:"  '+str(ti)+':')
# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 8

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960
print('MAE 8  sec  '+str(error)+':')
print('время при 8 секунде:"  '+str(ti)+':')
# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 9

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960
print('MAE 9  sec  '+str(error)+':')
print('время при 9 секунде:"  '+str(ti)+':')
# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()

step_predict = 10

forecast = []
results = []

for time in range(0,len(series) - window_size, step_predict):
     # print("hello")
     for i in range(step_predict):
         wnd = np.append(series[time+i:time + window_size], forecast[len(forecast)-i:])
        # wnd=series[time+i:time + window_size - i]
        #  print(time)
         forecast.append(model.predict(wnd[np.newaxis]))

# for i in range(step_predict):
#     forecast.append(model.predict(series[i:i + window_size][np.newaxis]))
#
# for time in range(step_predict, len(series) - window_size):
#     wnd = np.append(series[time+step_predict:time + window_size], forecast[len(forecast) - step_predict:])
#     forecast.append(model.predict(wnd[np.newaxis]))

# print(results)
# print(forecast)
forecast = forecast[:split_time]# -window_size:]
results = np.array(forecast)[:, 0, 0]

x_valid_1 = x_valid[20:]
time_valid_1 = time_valid[20:]
results_1 = results[:200]

error = tf.keras.metrics.mean_absolute_error(x_valid_1, results_1).numpy()
now1 = datetime.datetime.now()-now
ti=now1/960
print('MAE 10  sec  '+str(error)+':')
print('время при 10 секунде:"  '+str(ti)+':')
# plt.figure(figsize=(10, 6))
# start = 0
# end = None
# plt.plot(time_valid, x_valid, label='Input')
# plt.plot(time_valid_1, results_1, label='Forecasted')
# plt.title("LSTM, MAE = " + str(error))
# plt.xlabel("Time, s")
# plt.ylabel("Speed, m/s")
# plt.legend()
# plt.grid(True)
# plt.show()
