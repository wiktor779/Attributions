from src.utils import *
from src.first_and_last_touch_model import *

df = load_data()
df = create_touch_columns(df)

touch_methods = ['first_touch_utm_source',
                 'last_touch_utm_source',
                 'first_touch_utm_medium',
                 'last_touch_utm_medium']

for method in touch_methods:
    impact_dict = predict_channel_impact(df, method, True)
    print(f'{method}:\n {impact_dict}\n')
    visualize_channel_impact(impact_dict, method)

