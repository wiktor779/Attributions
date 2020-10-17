from math import sqrt
from sklearn.metrics import mean_squared_error


def measure_results(revenue_actual, revenue_predicted):
    if len(revenue_actual) != len(revenue_predicted):
        raise Exception("Length of inputs lists should be the same!")
    return sqrt(mean_squared_error(revenue_actual, revenue_predicted))