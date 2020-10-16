import pandas as pd
from collections import OrderedDict
from src.d00_utils.utils import get_unique_values, remove_channel_from_path, split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results
from src.d05_model_evaluation.measure_channels_impact import measure_channels_impact
from src.d07_visualisation.visualise_channel_impact import visualize_channel_impact


def last_touch_predict(df):
    means_for_each_channel = {}
    for channel in get_unique_values(df['utm_medium_list']):
        means_for_each_channel[channel] = df[df['utm_medium_list'].str[-1] == channel].revenue.mean()

    return df.apply(lambda x: means_for_each_channel[x['utm_medium_list'][-1]], axis=1).to_list()


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)

    revenue_predicted = last_touch_predict(conversion_paths_test)
    rmse = measure_results(conversion_paths_test.revenue, revenue_predicted)
    print(f'Last touch \tRMSE: {rmse}')

    channels_impact = measure_channels_impact(conversion_paths_test, last_touch_predict)
    visualize_channel_impact(channels_impact, 'last_touch')
    print(f'Channels impact in percentage {channels_impact}')

