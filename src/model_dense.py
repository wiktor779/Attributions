from src.utils import *
from src.first_and_last_touch_model import *
from keras.layers import Dense,LSTM,Embedding
from keras.models import Sequential,Model


df = load_data()
df = remove_outliers_z_score(df, 3.5)
df = transform_utm_columns_into_list_of_strings(df)
df = remove_none_entries(df)

df = create_utm_vectors(df, n=10)
x = df.utm_vector_source.to_list()
y = df.revenue.to_list()


def create_model_dense():
    model = Sequential()
    model.add(Dense(70, activation='relu', input_shape=(70,)))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['acc', 'mse'])
    return model

model = create_model_dense()
model.fit(x, y, batch_size=32, epochs=5)
predicted = model.predict(x)
rmse = measure_results(df.revenue, predicted)