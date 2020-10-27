import pandas as pd

VECTOR_LENGTH = 40
FILENAME = f'added_embedded_vectors_{VECTOR_LENGTH}'

ENCODING_DIC_UTM_SOURCE = {
    'google':       [1, 0, 0, 0],
    'google.com':   [1, 0, 0, 0],
    'facebook':     [0, 1, 0, 0],
    'facebook.com': [0, 1, 0, 0],
    'synerise':     [0, 0, 1, 0],
    'bing.com':     [0, 0, 0, 1],
}

ENCODING_DIC_UTM_MEDIUM = {
    'cpc':      [1, 0, 0, 0, 0, 0],
    'e-mail':   [0, 1, 0, 0, 0, 0],
    'organic':  [0, 0, 1, 0, 0, 0],
    'referral': [0, 0, 0, 1, 0, 0],
    'sms_text': [0, 0, 0, 0, 1, 0],
    'web_push': [0, 0, 0, 0, 0, 1],
}


def _fill_beginning_with_zeros(row, utm_column_name, encoding_dic, is_2d, is_time):
    path_length = len(row[utm_column_name])
    embedded_vector = []
    if path_length >= VECTOR_LENGTH:
        return embedded_vector
    embedded_touch = [0] * len(list(encoding_dic.values())[0])
    if is_time:
        embedded_touch.append(0)
    for i in range(VECTOR_LENGTH - path_length):
        if is_2d:
            embedded_vector.append(embedded_touch)
        else:
            embedded_vector.extend(embedded_touch)

    return embedded_vector


def transform_into_vector(row, utm_column_name, encoding_dic, is_2d, is_time):
    row[utm_column_name] = row[utm_column_name][-VECTOR_LENGTH:]
    row['days_till_conversions_list'] = row['days_till_conversions_list'][-VECTOR_LENGTH:]
    embedded_vector = _fill_beginning_with_zeros(row, utm_column_name, encoding_dic, is_2d, is_time)

    for i, touch in enumerate(row[utm_column_name]):
        embedded_touch = encoding_dic[touch].copy()
        if is_time:
            embedded_touch.append(row['days_till_conversions_list'][i])
        if is_2d:
            embedded_vector.append(embedded_touch)
        else:
            embedded_vector.extend(embedded_touch)

    return embedded_vector


if __name__ == '__main__':

    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')

    conversion_paths['utm_source_embedded_1d'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_source_list', ENCODING_DIC_UTM_SOURCE, False, False))
    conversion_paths['utm_source_embedded_1d_with_time'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_source_list', ENCODING_DIC_UTM_SOURCE, False, True))
    conversion_paths['utm_source_embedded_2d'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_source_list', ENCODING_DIC_UTM_SOURCE, True, False))
    conversion_paths['utm_source_embedded_2d_with_time'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_source_list', ENCODING_DIC_UTM_SOURCE, True, True))

    conversion_paths['utm_medium_embedded_1d'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_medium_list', ENCODING_DIC_UTM_MEDIUM, False, False))
    conversion_paths['utm_medium_embedded_1d_with_time'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_medium_list', ENCODING_DIC_UTM_MEDIUM, False, True))
    conversion_paths['utm_medium_embedded_2d'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_medium_list', ENCODING_DIC_UTM_MEDIUM, True, False))
    conversion_paths['utm_medium_embedded_2d_with_time'] = conversion_paths.apply(transform_into_vector, axis=1,
                                                    args=('utm_medium_list', ENCODING_DIC_UTM_MEDIUM, True, True))

    filepath = f'../../data/03_processed/{FILENAME}.pkl'
    conversion_paths.to_pickle(filepath)
    print(f'Saved file to: {filepath}')
