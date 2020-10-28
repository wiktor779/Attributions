import pandas as pd
import pickle
from src.d00_utils.utils import get_unique_values, split_data_into_train_and_test, remove_channel_from_path
from src.d05_model_evaluation.measure_results import measure_results
from src.d05_model_evaluation.measure_channels_impact import measure_channels_impact
from src.d07_visualisation.visualise_channel_impact import visualize_channel_impact

column_name = 'utm_source_list'
filepath = f'../../data/04_models/first_touch_{column_name}.pkl'


def first_touch_train(df, utm_column_name):
    means_for_each_channel = {}
    for channel in get_unique_values(df[utm_column_name]):
        means_for_each_channel[channel] = df[df[utm_column_name].str[0] == channel].revenue.mean()
    with open(filepath, 'wb') as handle:
        pickle.dump(means_for_each_channel, handle)


def first_touch_predict(df, utm_column_name):
    with open(filepath, 'rb') as handle:
        means_for_each_channel = pickle.load(handle)
    return df.apply(lambda x: means_for_each_channel[x[utm_column_name][0]], axis=1)


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)
    first_touch_train(conversion_paths_test, column_name)

    revenue_predicted = first_touch_predict(conversion_paths_test, column_name)
    rmse = measure_results(conversion_paths_test.revenue, revenue_predicted)
    print(f'First touch \tRMSE: {rmse}')

    channels_impact = measure_channels_impact(conversion_paths_test, first_touch_predict, column_name)
    visualize_channel_impact(channels_impact, f'first_touch_{column_name}')
    print(f'Channels impact in percentage {channels_impact}')
