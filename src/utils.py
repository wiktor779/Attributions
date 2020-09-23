import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


def load_data(path_to_file='../data/01_raw/conversion_paths.csv'):
    df = pd.read_csv(path_to_file)

    columns_ordered = [
        'client_uuid',
        'revenue',
        'utm_source',
        'utm_medium',
        'event_dates',
        'event_timestamps',
        'days_till_conversions',
        'conversion_date',
        'conversion_timestamp',
    ]
    return df[columns_ordered]


def transform_utm_columns_into_list_of_strings(df):
    def create_list_from_string(text):
        # "['word1', 'word2']" -> ['word1', 'word2']
        text = text[1:-1]
        words = text.split(',')
        return [word[1:-1] for word in words]

    df.utm_source = df.utm_source.apply(create_list_from_string)
    df.utm_medium = df.utm_medium.apply(create_list_from_string)
    return df


def get_unique_values(series):
    unique = set()
    for sources in series:
        for source in sources:
            unique.add(source)
    return unique


def get_number_of_occurrences(series):
    occurrences = {}
    occurrences = defaultdict(lambda:0,occurrences)
    for sources in series:
        for source in sources:
            occurrences[source] += 1
    return dict(occurrences)


def plot_number_of_occurrences(D):
    plt.rcParams["figure.figsize"]=20,10
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()))
    plt.show()


def remove_direct_entries(df):
    return df[(df.utm_source != "['(direct)']") & (df.utm_medium != "['(none)']")]


def remove_outliers_z_score(df, z=3.5):
    z_scores = stats.zscore(df.revenue)
    abs_z_scores = np.abs(z_scores)
    return df[abs_z_scores < z]


def measure_results(revenue_actual, revenue_predicted):
    if len(revenue_actual) != len(revenue_predicted):
        raise Exception("Length of inputs lists should be the same!")
    return sqrt(mean_squared_error(revenue_actual, revenue_predicted))


def predict_naive_results(df):
    length = df.shape[0]
    return [df.revenue.mean()]*length


ENCODING_DIC_SOURCE = {
    None:           [0, 0, 0, 0, 0, 0, 0],
    '(direct)':     [1, 0, 0, 0, 0, 0, 0],
    'bing.com':     [0, 1, 0, 0, 0, 0, 0],
    'facebook':     [0, 0, 1, 0, 0, 0, 0],
    'facebook.com': [0, 0, 0, 1, 0, 0, 0],
    'google':       [0, 0, 0, 0, 1, 0, 0],
    'google.com':   [0, 0, 0, 0, 0, 1, 0],
    'synerise':     [0, 0, 0, 0, 0, 0, 1],
}
ENCODING_DIC_MEDIUM = {
    None:       [0, 0, 0, 0, 0, 0, 0],
    '(none)':   [1, 0, 0, 0, 0, 0, 0],
    'cpc':      [0, 1, 0, 0, 0, 0, 0],
    'e-mail':   [0, 0, 1, 0, 0, 0, 0],
    'organic':  [0, 0, 0, 1, 0, 0, 0],
    'referral': [0, 0, 0, 0, 1, 0, 0],
    'sms_text': [0, 0, 0, 0, 0, 1, 0],
    'web_push': [0, 0, 0, 0, 0, 0, 1],
}


def crop_or_fill_to_length(utm_list, n):
    length = len(utm_list)
    if length > n:
        return utm_list[-n:]
    elif length < n:
        return [None]*(n-length) + utm_list
    else:
        return utm_list


def one_hot_encode_into_vector(utm_list, encoding_dic):
    vector = []
    for touch in utm_list:
        vector.extend(encoding_dic[touch])
    return vector


def transform_utm_into_vector(utm, encoding_dic, n):
    utm = utm.apply(crop_or_fill_to_length, args=(n,))
    utm = utm.apply(one_hot_encode_into_vector, args=(encoding_dic,))
    return utm


def create_utm_vectors(df, n=10):
    df['utm_vector_source'] = transform_utm_into_vector(df.utm_source, ENCODING_DIC_SOURCE, n)
    df['utm_vector_medium'] = transform_utm_into_vector(df.utm_medium, ENCODING_DIC_MEDIUM, n)
    return df

