import pandas as pd
from keras.models import load_model
from src.d00_utils.utils import split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results
from src.d05_model_evaluation.measure_channels_impact import measure_channels_impact
from src.d07_visualisation.visualise_channel_impact import visualize_channel_impact

model_name = 'lstm_model'


def lstm_predict(df):
    x_test = df['utm_medium_embedded_2d'].to_list()
    model_dense = load_model(f'../../data/04_models/{model_name}.h5')
    return model_dense.predict(x_test).ravel().tolist()


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/03_processed/added_embedded_vector.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)

    revenue_predicted = lstm_predict(conversion_paths_test)
    rmse = measure_results(conversion_paths_test.revenue, revenue_predicted)
    print(f'LSTM\tRMSE: {rmse}')

    channels_impact = measure_channels_impact(conversion_paths_test, lstm_predict)
    visualize_channel_impact(channels_impact, model_name)
    print(f'Channels impact in percentage {channels_impact}')
