from scipy import stats
import numpy as np
from src.d01_data.load_data import *


def _create_list_from_string(text):
    """ eg. "['word1', 'word2']" -> ['word1', 'word2']
    """
    text = text[1:-1]
    words = text.split(',')
    return [word[1:-1] for word in words]


def transform_utm_columns_into_list_of_strings(df):
    df['utm_source_list'] = df.utm_source.apply(_create_list_from_string)
    df['utm_medium_list'] = df.utm_medium.apply(_create_list_from_string)
    return df


def _remove_nones(path):
    return [touch for touch in path if touch != '(none)']


def remove_nones_from_conversions_path(df):
    # TODO: usuwać wszystkie pozostałe informacje (utm_source, timestamps itd) a nie tylko utm_medium_list
    df.utm_medium_list = df.utm_medium_list.apply(_remove_nones)
    return df


def remove_outliers_z_score(df, z=3.5):
    z_scores = stats.zscore(df.revenue)
    abs_z_scores = np.abs(z_scores)
    return df[abs_z_scores < z]


if __name__ == "__main__":
    conversion_paths = load_data()
    conversion_paths = transform_utm_columns_into_list_of_strings(conversion_paths)
    conversion_paths = remove_nones_from_conversions_path(conversion_paths)
    conversion_paths.to_pickle('../../data/02_intermediate/cleaned.pkl')
