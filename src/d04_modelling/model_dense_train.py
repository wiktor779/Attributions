import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from src.d00_utils.utils import split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results


def create_model_dense():
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(60,)))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['acc', 'mse'])
    return model


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/03_processed/added_embedded_vector.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)

    x_train = conversion_paths_train['utm_medium_embedded'].to_list()
    y_train = conversion_paths_train.revenue.to_list()
    x_test = conversion_paths_test['utm_medium_embedded'].to_list()
    y_test = conversion_paths_test.revenue.to_list()

    model_dense = create_model_dense()
    model_dense.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
    model_dense.save('../../data/04_models/basic_dense_model.h5')

    predicted_dense = model_dense.predict(x_test)
    rmse = measure_results(y_test, predicted_dense)
    print(f'Dense\tRMSE: {rmse}')
