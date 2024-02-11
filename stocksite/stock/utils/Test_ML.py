import pandas as pd
import numpy as np
# import data_handle as dh
from . import data_handle as dh

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation


def data_conversion(trade_code, lookback_days, pred_type='close'):
    dataframe_all = dh.handle_with_pandas()

    # dataframe_filtered = dataframe_all[dataframe_all['trade_code'] == 'ACI']
    dataframe_filtered = dataframe_all[dataframe_all['trade_code'] == trade_code]
    dataframe_filtered = dataframe_filtered.reset_index(drop=True).drop(columns=['trade_code'])

    dataframe_filtered['date'] = pd.to_datetime(dataframe_filtered['date'])
    dataframe_filtered['close'] = pd.to_numeric(dataframe_filtered['close'].str.replace(',', ''))
    dataframe_filtered['open'] = pd.to_numeric(dataframe_filtered['open'].str.replace(',', ''))
    dataframe_filtered['high'] = pd.to_numeric(dataframe_filtered['high'].str.replace(',', ''))
    dataframe_filtered['low'] = pd.to_numeric(dataframe_filtered['low'].str.replace(',', ''))

    dataframe_filtered.index = dataframe_filtered.pop('date')

    test_data = dataframe_filtered.tail(lookback_days)

    test_data.reset_index(inplace=True)
    test_data.drop(['volume', 'date'], axis=1, inplace=True)
    test_data = np.array([test_data.iloc[:, 0:4]])

    # dataframe_filtered['Target'] = dataframe_filtered['close'].shift(-1)
    dataframe_filtered['Target'] = dataframe_filtered[pred_type].shift(-1)
    dataframe_filtered.dropna(inplace=True)
    dataframe_filtered.reset_index(inplace=True)
    dataframe_filtered.drop(['volume', 'date'], axis=1, inplace=True)

    data_set = dataframe_filtered.iloc[:, 0:5]

    # sc = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = sc.fit_transform(data_set)

    scaled_data = np.array(data_set)

    X = []
    # lookback_days = 3

    for j in range(4):
        X.append([])
        for i in range(lookback_days, scaled_data.shape[0]):
            X[j].append(scaled_data[i - lookback_days:i, j])

    X = np.moveaxis(X, [0], [2])

    X, yi = np.array(X), np.array(scaled_data[lookback_days:, -1])

    y = np.reshape(yi, (len(yi), 1))

    # split_ratio = int(len(X) * 0.8)
    #
    # X_train, X_test = X[:split_ratio], X[split_ratio:]
    # y_train, y_test = y[:split_ratio], y[split_ratio:]

    return X, y, test_data


def train_and_predict_next_day(trade_code, lookback_days, pred_type):

    X, y, test_data = data_conversion(trade_code, lookback_days, pred_type)

    lstm_input = Input(shape=(lookback_days, 4), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(x=X, y=y, batch_size=4, epochs=3, shuffle=False, validation_split=0.1)

    y_pred = model.predict(test_data)

    return y_pred


if __name__ == '__main__':
    # dataframe_all = dh.handle_with_pandas()
    #
    # dataframe_filtered = dataframe_all[dataframe_all['trade_code'] == 'ACI']
    # dataframe_filtered = dataframe_filtered.reset_index(drop=True).drop(columns=['trade_code'])
    #
    # dataframe_filtered['date'] = pd.to_datetime(dataframe_filtered['date'])
    # dataframe_filtered['close'] = pd.to_numeric(dataframe_filtered['close'].str.replace(',', ''))
    # dataframe_filtered['open'] = pd.to_numeric(dataframe_filtered['open'].str.replace(',', ''))
    # dataframe_filtered['high'] = pd.to_numeric(dataframe_filtered['high'].str.replace(',', ''))
    # dataframe_filtered['low'] = pd.to_numeric(dataframe_filtered['low'].str.replace(',', ''))
    # dataframe_filtered['volume'] = pd.to_numeric(dataframe_filtered['volume'].str.replace(',', ''))
    #
    # dataframe_filtered.index = dataframe_filtered.pop('date')
    #
    # test_data = dataframe_filtered.tail(3)
    #
    # dataframe_filtered['Target'] = dataframe_filtered['close'].shift(-1)
    # dataframe_filtered.dropna(inplace=True)
    # dataframe_filtered.reset_index(inplace=True)
    #
    # test_data.reset_index(inplace=True)
    # test_data.drop(['volume', 'date'], axis=1, inplace=True)
    # test_data = np.array([test_data.iloc[:, 0:4]])
    #
    #
    # dataframe_filtered.drop(['volume', 'date'], axis=1, inplace=True)
    #
    # data_set = dataframe_filtered.iloc[:, 0:5]
    #
    # # sc = MinMaxScaler(feature_range=(0, 1))
    # # scaled_data = sc.fit_transform(data_set)
    #
    # scaled_data = np.array(data_set)
    #
    # X = []
    # lookback_days = 3
    #
    # for j in range(4):
    #     X.append([])
    #     for i in range(lookback_days, scaled_data.shape[0]):
    #         X[j].append(scaled_data[i-lookback_days:i, j])
    #
    # X = np.moveaxis(X, [0], [2])
    #
    # X, yi = np.array(X), np.array(scaled_data[lookback_days:, -1])
    #
    # y = np.reshape(yi, (len(yi), 1))
    #
    # split_ratio = int(len(X) * 0.8)
    #
    # X_train, X_test = X[:split_ratio], X[split_ratio:]
    # y_train, y_test = y[:split_ratio], y[split_ratio:]
    # #
    # # model = keras.Sequential([
    # #     InputLayer(input_shape=(lookback_days, 4), name='lstm_input'),
    # #     LSTM(150),
    # #     Dense(16, activation='relu'),
    # #     Dense(1)
    # # ])
    # #
    # # model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    # # model.fit(X_train, y_train, epochs=10)
    #
    # lstm_input = Input(shape=(lookback_days, 4), name='lstm_input')
    # inputs = LSTM(150, name='first_layer')(lstm_input)
    # inputs = Dense(1, name='dense_layer')(inputs)
    # output = Activation('linear', name='output')(inputs)
    # model = Model(inputs=lstm_input, outputs=output)
    # model.compile(optimizer=Adam(), loss='mse')
    # model.fit(x=X_train, y=y_train, batch_size=4, epochs=100, shuffle=False, validation_split=0.1)
    #
    # y_pred = model.predict(test_data)
    #
    # print(y_pred)

    pred = train_and_predict_next_day('ACI', 2, 'open')
    print(pred)
