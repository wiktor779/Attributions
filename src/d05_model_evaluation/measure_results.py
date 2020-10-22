from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def measure_results(revenue_actual, revenue_predicted, mae=False):
    if len(revenue_actual) != len(revenue_predicted):
        raise Exception("Length of inputs lists should be the same!")
    if mae:
        return sqrt(mean_absolute_error(revenue_actual, revenue_predicted))
    return sqrt(mean_squared_error(revenue_actual, revenue_predicted))
