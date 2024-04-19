import numpy as np

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

data = {'date': pd.date_range(start='2022-01-01', periods=100),
        'force': np.random.rand(100)}

df = pd.DataFrame(data)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['force'].values.reshape(-1, 1))


def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)


time_steps = 10
X, y = create_dataset(scaled_data, time_steps)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=100, batch_size=32)


def predict_force(model, date):
    # Assuming 'date' is in the format 'YYYY-MM-DD'
    input_date = pd.to_datetime(date)
    last_date = df['date'].max()
    prediction_date = last_date + pd.DateOffset(days=1)
    while prediction_date <= input_date:
        # Prepare input data for prediction
        last_values = df['force'].values[-time_steps:]
        input_data = np.array(last_values).reshape(1, time_steps, 1)
        # Predict force for the next day
        predicted_force = model.predict(input_data)
        # Update data for the next prediction
        df.loc[len(df)] = [prediction_date, predicted_force[0][0]]
        prediction_date += pd.DateOffset(days=1)
    return df[df['date'] == input_date]['force'].values[0]

# Example usage
user_input_date = '2024-04-15'
predicted_force = predict_force(model, user_input_date)
print(f'Predicted force on {user_input_date}: {predicted_force}')
