from scipy import stats
import numpy as np
from src.d00_utils.utils import remove_channel_from_path
from src.d01_data.load_data import load_data


def _create_list_from_string(text):
    """ eg. "['word1', 'word2']" -> ['word1', 'word2']
    """
    text = text[1:-1]
    words = text.split(',')
    return [word[1:-1] for word in words]


def transform_utm_columns_into_list(df):
    columns_to_transform = \
        ['utm_medium', 'utm_source', 'event_dates', 'event_timestamps']
    for column in columns_to_transform:
        df[f'{column}_list'] = df[column].apply(_create_list_from_string)
    df['days_till_conversions_list'] = df['days_till_conversions'].apply(lambda row: list(map(int, row[1:-1].split(','))))
    return df


def remove_outliers_z_score(df, z=3.5):
    z_scores = stats.zscore(df.revenue)
    abs_z_scores = np.abs(z_scores)
    return df[abs_z_scores < z]


if __name__ == "__main__":
    conversion_paths = load_data()
    conversion_paths = remove_outliers_z_score(conversion_paths, 3.5)
    conversion_paths = transform_utm_columns_into_list(conversion_paths)
    conversion_paths = remove_channel_from_path(conversion_paths, '(none)')
    filepath = '../../data/02_intermediate/cleaned.pkl'
    conversion_paths.to_pickle(filepath)
    print(f'Saved file to: {filepath}')
