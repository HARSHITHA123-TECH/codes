import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
import time
import schedule

schedule.clear()

df = pd.read_csv("AEP_hourly.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
dataset = df.set_index("Datetime")
NewDataSet = dataset.resample('D').mean()
Training_Set = NewDataSet.iloc[:, 0:1]
sc = MinMaxScaler(feature_range=(0, 1))
Train = sc.fit_transform(Training_Set)

class PicklableModel:
    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        return self.model.to_json()

    def __setstate__(self, state):
        self.model = tf.keras.models.model_from_json(state)

    def predict(self, input_data):
        return self.model.predict(input_data)

def train():
    X_Train = []
    Y_Train = []
    timesteps = 60  # Example timestep for sequence data
    
    for i in range(timesteps, Train.shape[0]):
        X_Train.append(Train[i-timesteps:i, 0])
        Y_Train.append(Train[i, 0])

    X_Train = np.array(X_Train)
    Y_Train = np.array(Y_Train)
    X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], 1))  # reshape for LSTM

    def build_lstm_model(input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_lstm_model((X_Train.shape[1], 1))
    model.fit(X_Train, Y_Train, epochs=1, batch_size=32)
    picklable_model = PicklableModel(model)

    with open(r'C:\Pickle\model.pkl', 'wb') as to_file:
        pickle.dump(picklable_model, to_file)

    with open(r'C:\Pickle\model.pkl', 'rb') as f:
        picklable_model = pickle.load(f)

schedule.every(1).minutes.do(train)

try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("Training stopped.")

#app.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
from flask import Flask, render_template, request
import time
import schedule

# Define the PicklableModel class directly in this script
class PicklableModel:
    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        return self.model.to_json()

    def __setstate__(self, state):
        self.model = tf.keras.models.model_from_json(state)

    def predict(self, input_data):
        return self.model.predict(input_data)


with open(r'C:\Pickle\model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

df1 = pd.read_csv("AEP_hourly1.csv")
dataset = df1
dataset["Month"] = pd.to_datetime(df1["Datetime"]).dt.month
dataset["Year"] = pd.to_datetime(df1["Datetime"]).dt.year
dataset["Date"] = pd.to_datetime(df1["Datetime"]).dt.date
dataset["Time"] = pd.to_datetime(df1["Datetime"]).dt.time
dataset["Week"] = pd.to_datetime(df1["Datetime"]).dt.isocalendar().week
dataset["Day"] = pd.to_datetime(df1["Datetime"]).dt.day_name()
dataset = dataset.set_index("Datetime")
dataset.index = pd.to_datetime(dataset.index)
dataset.head(1)
Training_Set = dataset.iloc[:,0:1]
# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
para1_scaled = scaler.fit_transform(Training_Set)

X_para1 = []
for i in range(60, len(para1_scaled)):
    X_para1.append(para1_scaled[i-60:i])
X_para1 = np.array(X_para1)

X_para1 = np.reshape(X_para1, (X_para1.shape[0], X_para1.shape[1], 1))

num_predictions = len(df1) - 60  # Number of predictions = total length - sequence length
predicted_values = loaded_model.predict(X_para1[:num_predictions])


predicted_values = scaler.inverse_transform(predicted_values)
predicted_values=predicted_values.reshape(-1,1)
dates = df1.index[60:].to_list()
True_MegaWatt = df1["AEP_MW"].iloc[60:].to_list()


Predicted_MegaWatt = [x[0] for x in predicted_values]
dates = dates[:len(Predicted_MegaWatt)]
True_MegaWatt = True_MegaWatt[:len(Predicted_MegaWatt)]

Machine_Df = pd.DataFrame(data={
    "Date": dates,
    "TrueMegaWatt": True_MegaWatt,
    "PredictedMegaWatt": Predicted_MegaWatt
})
print(predicted_values)

