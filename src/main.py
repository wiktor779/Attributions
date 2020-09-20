from src.utils import *

df = load_data()
df = remove_direct_entries(df, 'medium')
df = remove_outliers_z_score(df, 3.5)
predicted = predict_naive_results(df)
rmse_naive = measure_results(df.revenue, predicted)
print(f'Naive solution always predicting revenue = {df.revenue.mean()}, RMSE = {rmse_naive}')
