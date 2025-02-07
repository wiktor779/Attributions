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


def _unify_source_names(path):
    return [source.replace('.com', '') for source in path]


if __name__ == "__main__":
    conversion_paths = load_data()
    conversion_paths = remove_outliers_z_score(conversion_paths, 3.5)
    conversion_paths = transform_utm_columns_into_list(conversion_paths)
    conversion_paths.to_pickle('../../data/02_intermediate/converted_into_lists.pkl')
    conversion_paths = remove_channel_from_path(conversion_paths, '(none)', 'utm_medium_list')
    conversion_paths = remove_channel_from_path(conversion_paths, '(direct)', 'utm_source_list')
    conversion_paths.to_pickle('../../data/02_intermediate/removed_direct_entries.pkl')
    conversion_paths['utm_source_list'] = conversion_paths['utm_source_list'].apply(_unify_source_names)
    conversion_paths.to_pickle('../../data/02_intermediate/cleaned.pkl')
    print(f'Saved files to ../../data/02_intermediate')
