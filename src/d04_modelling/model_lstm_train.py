import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from src.d00_utils.utils import split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results

model_name = 'lstm_model'


def create_model_lstm():
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(10, 6)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='rmsprop', loss="mse", metrics=['mse', 'mae'])
    return model


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/03_processed/added_embedded_vector.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)

    column_name = 'utm_medium_embedded_2d'
    x_train = conversion_paths_train[column_name].to_list()
    y_train = conversion_paths_train.revenue.to_list()
    x_test = conversion_paths_test[column_name].to_list()
    y_test = conversion_paths_test.revenue.to_list()

    model_lstm = create_model_lstm()
    model_lstm.fit(x_train, y_train, epochs=10, batch_size=32, validation_data = (x_test, y_test))
    filepath = f'../../data/04_models/{model_name}.h5'
    model_lstm.save(filepath)
    print(f'Saved file to: {filepath}')

    predicted_dense = model_lstm.predict(x_test)
    rmse = measure_results(y_test, predicted_dense)
    print(f'LSTM\tRMSE: {rmse}')
