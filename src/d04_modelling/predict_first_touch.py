import pandas as pd
from src.d00_utils.utils import get_unique_values, split_data_into_train_and_test, remove_channel_from_path
from src.d05_model_evaluation.measure_results import measure_results
from src.d05_model_evaluation.measure_channels_impact import measure_channels_impact
from src.d07_visualisation.visualise_channel_impact import visualize_channel_impact


def predict_first_touch(df):
    means_for_each_channel = {}
    for channel in get_unique_values(df['utm_medium_list']):
        means_for_each_channel[channel] = df[df['utm_medium_list'].str[0] == channel].revenue.mean()

    return df.apply(lambda x: means_for_each_channel[x['utm_medium_list'][0]], axis=1).to_list()


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)
    revenue_predicted_first_touch = predict_first_touch(conversion_paths_test)
    rmse = measure_results(conversion_paths_test.revenue, revenue_predicted_first_touch)
    print(f'First touch \tRMSE: {rmse}')

    channels_impact = measure_channels_impact(conversion_paths_test, predict_first_touch)
    visualize_channel_impact(channels_impact, 'first_touch')
    print(channels_impact)
