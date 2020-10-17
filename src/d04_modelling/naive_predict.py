import pandas as pd
from src.d00_utils.utils import split_data_into_train_and_test
from src.d05_model_evaluation.measure_results import measure_results


def predict_naive(train, length):
    revenue_mean = train.revenue.mean()
    return [revenue_mean]*length


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    conversion_paths_train, conversion_paths_test = split_data_into_train_and_test(conversion_paths)
    revenue_predicted_naive = predict_naive(conversion_paths_train, conversion_paths_test.shape[0])
    rmse = measure_results(conversion_paths_test.revenue, revenue_predicted_naive)
    print(f'Naive \tRMSE: {rmse}')
