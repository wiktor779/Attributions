from src.utils import *


def get_first_touch(touches):
    direct = ['(direct)', '(none)']
    for touch in touches:
        if touch not in direct:
            return touch
    return None


def get_last_touch(touches):
    direct = ['(direct)', '(none)']
    for touch in reversed(touches):
        if touch not in direct:
            return touch
    return None


def create_touch_columns(df):
    df['first_touch_utm_source'] = df['utm_source'].apply(get_first_touch)
    df['last_touch_utm_source'] = df['utm_source'].apply(get_last_touch)
    df['first_touch_utm_medium'] = df['utm_medium'].apply(get_first_touch)
    df['last_touch_utm_medium'] = df['utm_medium'].apply(get_last_touch)
    return df


def transform_into_percentage_values(revenue_dict, precision=2):
    revenue_sum = sum(revenue_dict.values())
    for k in revenue_dict:
        revenue_dict[k] = round(revenue_dict[k] / revenue_sum, precision)
    return revenue_dict


def predict_channel_impact(df, utm, percentage=False):
    df = df[df[utm].notna()]
    revenue_sum_for_each_channel = {}
    for channel in get_unique_values(df[utm]):
        revenue_sum_for_each_channel[channel] = df[df[utm] == channel].revenue.sum()
    if percentage:
        return transform_into_percentage_values(revenue_sum_for_each_channel)
    return revenue_sum_for_each_channel

