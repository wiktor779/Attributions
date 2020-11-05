import pandas as pd
from keras.models import load_model
from src.d00_utils.utils import split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results
from src.d05_model_evaluation.measure_channels_impact import measure_channels_impact
from src.d07_visualisation.visualise_channel_impact import visualize_channel_impact

vector_length = 5
column_name = 'utm_medium_embedded_2d_with_time'
model_name = f'dense_{column_name}__vec_length_{vector_length}'


def lstm_predict(df, utm_column_name):
    x_test = df[column_name].to_list()
    model_dense = load_model(f'../../data/04_models/{model_name}.h5')
    return model_dense.predict(x_test).ravel().tolist()


if __name__ == "__main__":
    conversion_paths = pd.read_pickle(f'../../data/03_processed/added_embedded_vectors_{vector_length}.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)

    revenue_predicted = lstm_predict(conversion_paths_test, column_name)
    rmse = measure_results(conversion_paths_test.revenue, revenue_predicted)
    print(f'LSTM\tRMSE: {rmse}')

    channels_impact = measure_channels_impact(conversion_paths_test, lstm_predict, f'{column_name[:10]}_list')
    visualize_channel_impact(channels_impact, model_name)
    print(f'Channels impact in percentage {channels_impact}')
