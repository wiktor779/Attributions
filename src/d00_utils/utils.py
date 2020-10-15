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


def _remove_from_path(path, utm_to_remove):
    return [touch for touch in path if touch != utm_to_remove]


def remove_utm_from_path(df, utm_to_remove):
    # TODO: usuwać wszystkie pozostałe informacje (utm_source, timestamps itd) a nie tylko utm_medium_list
    df.utm_medium_list = df.utm_medium_list.apply(_remove_from_path, args=(utm_to_remove,))
    df = df[df.utm_medium_list.map(len) > 0]
    return df



































