from collections import Counter
import pandas as pd


def save_to_file_utm_source_and_medium_pairs_occurrence(df, filename):
    pairs = []
    for index, row in df.iterrows():
        for index_utm in range(len(row.utm_source_list)):
            pair = (row.utm_source_list[index_utm], row.utm_medium_list[index_utm])
            pairs.append(pair)

    c = Counter(pairs)
    filepath = f'../../results/{filename}.txt'
    with open(filepath, 'w') as f:
        for k, v in c.most_common():
            f.write(f'{k} {v}\n')
    print(f'Saved file to: {filepath}')
    return c


if __name__ == '__main__':
    conversion_paths = pd.read_pickle('../../data/02_intermediate/converted_into_lists.pkl')
    save_to_file_utm_source_and_medium_pairs_occurrence(conversion_paths, 'utm_source_and_medium_pairs_occurrence_raw')
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    save_to_file_utm_source_and_medium_pairs_occurrence(conversion_paths, 'utm_source_and_medium_pairs_occurrence_cleaned')
