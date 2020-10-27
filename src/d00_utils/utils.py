from collections import defaultdict


def get_unique_values(paths):
    unique = set()
    for path in paths:
        for touch in path:
            unique.add(touch)
    return unique


def get_number_of_occurrences(series):
    occurrences = {}
    occurrences = defaultdict(lambda: 0, occurrences)
    for sources in series:
        for source in sources:
            occurrences[source] += 1
    return dict(occurrences)


def split_data_into_train_and_test(df):
    df_test = df.iloc[::5, :]
    df_train = df.drop(df_test.index)
    return df_train, df_test


def _remove_from_path(row, utm_to_remove, columns_to_clean, utm_column_name):
    indexes_to_be_removed = [i for i in range(len(row[utm_column_name])) if row[utm_column_name][i] == utm_to_remove]
    for column in columns_to_clean:
        for index in sorted(indexes_to_be_removed, reverse=True):
            del row[column][index]


def remove_channel_from_path(df, utm_to_remove, utm_column_name):
    df = df.copy()
    columns_to_clean = ['utm_medium_list', 'utm_source_list', 'days_till_conversions_list', 'event_dates_list',
                        'event_timestamps_list']
    df.apply(_remove_from_path, axis=1, args=(utm_to_remove, columns_to_clean, utm_column_name))
    df = df[df.utm_medium_list.map(len) > 0]
    return df






























