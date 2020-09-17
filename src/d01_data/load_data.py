import pandas as pd


def load_data(path_to_file):
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
