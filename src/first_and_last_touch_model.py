from src.utils import *


def create_first_or_last_touch_column(df, first_or_last, utm):
    if first_or_last == 'first_touch':
        return df[utm].str[0]
    elif first_or_last == 'last_touch':
        return df[utm].str[-1]
    else:
        raise Exception('set first_or_last to \'first_touch\' or \'last touch\'')


def predict_touch_model(df, first_or_last, utm):
    df['touch'] = create_first_or_last_touch_column(df, first_or_last, utm)
    means_for_each_channel = {}
    for channel in get_unique_values(df[utm]):
        means_for_each_channel[channel] = df[df['touch'] == channel].revenue.mean()

    return df.apply(lambda x: means_for_each_channel[x.touch], axis=1).to_list()
