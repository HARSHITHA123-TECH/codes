# codes
code
energy consumption prediction
import tensorflow as tf
import pickle

# Custom pickling logic
class PicklableModel:
    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        return self.model.to_json()

    def __setstate__(self, state):
        self.model = tf.keras.models.model_from_json(state)

def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))

    # Adding the output layer
    model.add(tf.keras.layers.Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Assuming X_Train, Y_Train have been defined elsewhere
input_shape = (X_Train.shape[1], 1)

# Build the model
regressor = build_lstm_model(input_shape)

# Train the model
regressor.fit(X_Train, Y_Train, epochs=2, batch_size=32)

# Wrap your model with the PicklableModel class
picklable_model = PicklableModel(regressor)

# Now you can pickle the picklable_model
with open('C://Pickle//model.pkl', 'wb') as to_file:
    pickle.dump(picklable_model, to_file)
import pickle

# Load the pickled model
with open('C://Pickle//model.pkl', 'rb') as file:
    pickled_model = pickle.load(file)

# Access the original model
loaded_model = pickled_model.model

# Now you can use the loaded_model for prediction or further training
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
loaded_model = pickled_model.model

# Read the CSV file
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
print(X_para1)
# Reshape the input data to match the model's input shape
X_para1 = np.reshape(X_para1, (X_para1.shape[0], X_para1.shape[1], 1))
# Only predict values for which you have corresponding true values
num_predictions = len(df1) - 60  # Number of predictions = total length - sequence length
predicted_values = loaded_model.predict(X_para1[:num_predictions])

# Inverse transform the predictions to get the original scale
predicted_values = scaler.inverse_transform(predicted_values)

# Update dates and true values accordingly
# dates = df1.index[60:].to_list()
# True_MegaWatt = df1["AEP_MW"].iloc[60:].to_list()


Predicted_MegaWatt = [x[0] for x in predicted_values]
dates = dates[:len(Predicted_MegaWatt)]
True_MegaWatt = True_MegaWatt[:len(Predicted_MegaWatt)]

Machine_Df = pd.DataFrame(data={
    "Date": dates,
    "TrueMegaWatt": True_MegaWatt,
    "PredictedMegaWatt": Predicted_MegaWatt
})
##app.py
# app.py

from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

app = Flask(__name__)

# Load the pickled model
with open('C://Pickle//model.pkl', 'rb') as file:
    pickled_model = pickle.load(file)

# Access the original model
loaded_model = pickled_model.model

@app.route('/')
def index():
    return "This is the main page."

@app.route('/predict')
def predict():
    # Read the CSV file
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
    print(X_para1)
    # Reshape the input data to match the model's input shape
    X_para1 = np.reshape(X_para1, (X_para1.shape[0], X_para1.shape[1], 1))
    # Only predict values for which you have corresponding true values
    num_predictions = len(df1) - 60  # Number of predictions = total length - sequence length
    predicted_values = loaded_model.predict(X_para1[:num_predictions])

    # Inverse transform the predictions to get the original scale
    predicted_values = scaler.inverse_transform(predicted_values)

    # Update dates and true values accordingly
    dates = df1.index[60:].to_list()
    True_MegaWatt = df1["AEP_MW"].iloc[60:].to_list()

    Predicted_MegaWatt = [x[0] for x in predicted_values]

    Machine_Df = pd.DataFrame(data={
        "Date": dates,
        "TrueMegaWatt": True_MegaWatt,
        "PredictedMegaWatt": Predicted_MegaWatt
    })

    # Render the result.html template with the Machine_Df DataFrame
    return render_template('result.html', machine_df=Machine_Df)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
## results.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
</head>
<body>
    <h1>Prediction Results</h1>
    <table border ="1">
        <thead>
            <tr>
                <th>Date</th>
                <th>True MegaWatt</th>
                <th>Predicted MegaWatt</th>
            </tr>
        </thead>
        <tbody>
            {% for index, row in machine_df.iterrows() %}
            <tr>
                <td>{{ row['Date'] }}</td>
                <td>{{ row['TrueMegaWatt'] }}</td>
                <td>{{ row['PredictedMegaWatt'] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>

