import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Загрузка данных
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USDT&limit=1000')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')

# Подготовка данных
target_col = 'close'
window_len = 10
test_size = 0.1
zero_base = True

def extract_window_data(df, window_len, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = tmp / tmp.iloc[0] - 1
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, target_col, window_len, zero_base=True, test_size=0.1):
    train_data, test_data = train_test_split(df, test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

train, test, X_train, X_test, y_train, y_test = prepare_data(hist, target_col, window_len, zero_base, test_size)

# Создание модели
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Обучение модели
epochs = 50
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test), shuffle=True)

# Предсказание
preds = model.predict(X_test).squeeze()

# Восстановление исходных значений
if zero_base:
    preds = test[target_col].values[:-window_len] * (preds + 1)
    y_test = test[target_col].values[:-window_len] * (y_test + 1)

# График
def line_plot(line1, line2, label1=None, label2=None, title='', lw=1.5):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    ax.grid(True)

plt.style.use('ggplot')
line_plot(y_test, preds, 'actual', 'prediction', lw=1.5)
plt.show()

# Предсказание на текущую дату
latest_data = hist[-window_len:]
latest_window = extract_window_data(latest_data, window_len, zero_base)
latest_prediction = model.predict(latest_window).squeeze()

latest_price = hist[target_col].values[-1]
predicted_price = latest_price * (latest_prediction + 1)
print("Предсказанная цена на текущую дату:", predicted_price)
