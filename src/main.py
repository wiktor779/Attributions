from src.utils import *
from src.first_and_last_touch_model import *

df = load_data()
df = transform_utm_columns_into_list_of_strings(df)
df = remove_none_entries(df)
df = create_touch_columns(df)
# save_to_file_utm_source_and_medium_pairs_occurrence(df)  # take long time

touch_methods = [
                # 'first_touch_utm_source',
                #  'last_touch_utm_source',
                 'first_touch_utm_medium',
                 'last_touch_utm_medium']

for method in touch_methods:
    # impact_dict = predict_channel_impact(df, method, True)
    predicted = predict_naive(df, method)
    # print(f'{method}:\n {impact_dict}\n')
    print(f'RMSE: {measure_results(df.revenue, predicted)}')
    # visualize_channel_impact(impact_dict, method)

