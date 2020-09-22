from src.utils import *
from src.first_and_last_touch_model import *

df = load_data()
df = remove_direct_entries(df)
df = remove_outliers_z_score(df, 3.5)

df = transform_utm_columns_into_list_of_strings(df)

predicted = predict_naive_results(df)
rmse = measure_results(df.revenue, predicted)
print(f'Naive solution always predicting revenue = {df.revenue.mean()}, RMSE = {rmse}')

predicted = predict_touch_model(df, 'first_touch', 'utm_medium')
rmse = measure_results(df.revenue, predicted)
print(f'Naive solution first_touch, utm_medium, RMSE = {rmse}')

predicted = predict_touch_model(df, 'last_touch', 'utm_medium')
rmse = measure_results(df.revenue, predicted)
print(f'Naive solution last_touch, utm_medium, RMSE = {rmse}')

predicted = predict_touch_model(df, 'first_touch', 'utm_source')
rmse = measure_results(df.revenue, predicted)
print(f'Naive solution first_touch, utm_source, RMSE = {rmse}')

predicted = predict_touch_model(df, 'last_touch', 'utm_source')
rmse = measure_results(df.revenue, predicted)
print(f'Naive solution last_touch, utm_source, RMSE = {rmse}')



