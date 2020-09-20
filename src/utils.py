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


def remove_direct_entries(df, utm):
    if utm == 'medium':
        return df[(df.utm_medium != "['(none)']")]
    elif utm == 'source':
        return df[(df.utm_medium != "['(direct)']")]
    else:
        raise Exception("You need to set utm value ('source' or 'medium')")


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
