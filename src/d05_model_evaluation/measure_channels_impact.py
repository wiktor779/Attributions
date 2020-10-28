from collections import OrderedDict
import pickle
from src.d00_utils.utils import remove_channel_from_path, get_unique_values


def _to_percentage(channel_impact_dict):
    summed = sum(channel_impact_dict.values())
    for channel in channel_impact_dict:
        channel_impact_dict[channel] = round(channel_impact_dict[channel] / summed * 100, 2)
    return channel_impact_dict


def _deep_copy(df):
    """ I made a mistake at beginning of the project by inserting mutable object into DataFrame which is anti-patterns.
    Because i don't have time to correct it, this is workaround """
    return pickle.loads(pickle.dumps(df))


def measure_channels_impact(df, predict_revenue_func, utm_column_name):
    channels_revenue_dict = {}
    initial_revenue_sum = sum(df.revenue)
    channels = get_unique_values(df[utm_column_name])
    for channel in channels:
        df_copy = _deep_copy(df)
        conversion_paths_test_without_channel = remove_channel_from_path(df_copy, channel, utm_column_name)
        revenue_predicted_without_channel = predict_revenue_func(conversion_paths_test_without_channel, utm_column_name)
        channels_revenue_dict[channel] = initial_revenue_sum - sum(revenue_predicted_without_channel)
    channels_revenue_dict = OrderedDict(sorted(channels_revenue_dict.items()))
    channels_revenue_dict = _to_percentage(channels_revenue_dict)
    return channels_revenue_dict


