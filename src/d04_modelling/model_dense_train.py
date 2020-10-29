import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from src.d00_utils.utils import split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results

vector_length = 40
column_name = 'utm_source_embedded_1d_with_time'
model_name = f'dense_{column_name}__vec_length_{vector_length}'


def create_model_dense(input_length):
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(input_length,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['mse', 'mae'])
    return model


if __name__ == "__main__":
    conversion_paths = pd.read_pickle(f'../../data/03_processed/added_embedded_vectors_{vector_length}.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)

    x_train = conversion_paths_train[column_name].to_list()
    y_train = conversion_paths_train.revenue.to_list()
    x_test = conversion_paths_test[column_name].to_list()
    y_test = conversion_paths_test.revenue.to_list()

    model_dense = create_model_dense(len(x_train[0]))
    model_dense.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
    filepath = f'../../data/04_models/{model_name}.h5'
    model_dense.save(filepath)
    print(f'Saved file to: {filepath}')

    predicted_dense = model_dense.predict(x_test)
    rmse = measure_results(y_test, predicted_dense)
    print(f'Dense\tRMSE: {rmse}')
