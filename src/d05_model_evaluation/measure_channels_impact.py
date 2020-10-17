from collections import OrderedDict
from src.d00_utils.utils import remove_channel_from_path, get_unique_values


def _to_percentage(channel_impact_dict):
    summed = sum(channel_impact_dict.values())
    for channel in channel_impact_dict:
        channel_impact_dict[channel] = round(channel_impact_dict[channel] / summed * 100, 2)
    return channel_impact_dict


def measure_channels_impact(df, predict_revenue_func):
    channels_revenue_dict = {}
    initial_revenue_sum = sum(df.revenue)
    channels = get_unique_values(df['utm_medium_list'])
    for channel in channels:
        conversion_paths_test_without_channel = remove_channel_from_path(df, channel)
        revenue_predicted_without_channel = predict_revenue_func(conversion_paths_test_without_channel)
        channels_revenue_dict[channel] = initial_revenue_sum - sum(revenue_predicted_without_channel)
    channels_revenue_dict = OrderedDict(sorted(channels_revenue_dict.items()))
    channels_revenue_dict = _to_percentage(channels_revenue_dict)
    return channels_revenue_dict


