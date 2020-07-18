import pandas as pd


def load_data():
    df = pd.read_csv('../../data/01_raw/conversion_paths.csv')

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
