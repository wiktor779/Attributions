import pandas as pd

ENCODING_DIC_UTM_MEDIUM = {
    'cpc':      [1, 0, 0, 0, 0, 0],
    'e-mail':   [0, 1, 0, 0, 0, 0],
    'organic':  [0, 0, 1, 0, 0, 0],
    'referral': [0, 0, 0, 1, 0, 0],
    'sms_text': [0, 0, 0, 0, 1, 0],
    'web_push': [0, 0, 0, 0, 0, 1],
}


def one_hot_encode_into_vector(path, encoding_dic, vector_length):
    fill_to_desirable_length = len(encoding_dic) * (vector_length - len(path))
    vector = [0] * fill_to_desirable_length
    for touch in path:
        vector.extend(encoding_dic[touch])
    return vector


def transform_utm_into_vector(paths, encoding_dic, vector_length):
    """ takes vector_length last touches in path """
    paths_cropped = paths.apply(lambda path: path[-vector_length:])
    paths_embedded_vector = paths_cropped.apply(one_hot_encode_into_vector, args=(encoding_dic, vector_length))
    return paths_embedded_vector


if __name__ == '__main__':
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    conversion_paths['utm_medium_embedded'] = transform_utm_into_vector(
        conversion_paths.utm_medium_list, ENCODING_DIC_UTM_MEDIUM, 10)
    filepath = '../../data/03_processed/added_embedded_vector.pkl'
    conversion_paths.to_pickle(filepath)
    print(f'Saved file to: {filepath}')
