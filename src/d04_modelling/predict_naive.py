import pandas as pd
from src.d01_data.load_data import load_data
from src.d05_model_evaluation.measure_results import measure_results


def predict_naive(df):
    revenue_mean = df.revenue.mean()
    length = df.shape[0]
    return [revenue_mean]*length


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    revenue_predicted_naive = predict_naive(conversion_paths)
    rmse = measure_results(conversion_paths.revenue, revenue_predicted_naive)
    print(f'RMSE: {rmse}')
