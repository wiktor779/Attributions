import pandas as pd
from src.d00_utils.utils import get_unique_values
from src.d05_model_evaluation.measure_results import measure_results


def predict_first_touch(df):
    means_for_each_channel = {}
    for channel in get_unique_values(df['utm_medium_list']):
        means_for_each_channel[channel] = df[df['utm_medium_list'].str[-1] == channel].revenue.mean()

    return df.apply(lambda x: means_for_each_channel[x['utm_medium_list'][-1]], axis=1).to_list()


if __name__ == "__main__":
    conversion_paths = pd.read_pickle('../../data/02_intermediate/cleaned.pkl')
    revenue_predicted_last_touch = predict_first_touch(conversion_paths)
    rmse = measure_results(conversion_paths.revenue, revenue_predicted_last_touch)
    print(f'Last touch \tRMSE: {rmse}')
