import subprocess
import sys

import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# Install pandas from within a Python script
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

import pandas as pd

def get_feature_and_target(data_frame, column_of_interest):    
 
    # Extract the column of interest
    df = data_frame[column_of_interest].to_frame()
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)
    # Create feature colums
    df[f"{column_of_interest}_L1"] = df[column_of_interest].shift(1)
    # Drop NaN values
    df = df.dropna()
    # Define the target and features
    target = df[column_of_interest]
    features = df.drop(columns=[column_of_interest])
    
    return target, features

# Function to predixt the "humidity", "wind_speed", "mean_pressure" columns
def pacf_plot(df,col):
    y_train = df[col]
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_pacf(y_train, ax = ax)

    plt.xlabel("Lag [Days]")
    plt.ylabel("Correlation Coefficient")
    plt.title(f"{col} partial autocorrelation function")

# Perform a walk forward validation with Autoregressive model
def autoregressive_model(train_df,test_df, col, lag = None):
    # Set the frequency explicitly to daily ('D')
    train_df = train_df.asfreq('D')
    test_df = test_df.asfreq('D')

    # Convert to Pandas Series
    y_train = train_df[col].squeeze()
    y_test =test_df[col].squeeze()

    # Change the index to datetime
    y_train.index = pd.to_datetime(y_train.index)
    y_test.index = pd.to_datetime(y_test.index)


    #Build a baseline model
    y_train_mean = y_train.mean()
    y_pred_baseline = [y_train_mean] * len(y_train)
    mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

    print(f"{col} Mean Absolute Error of Baseline Model: {mae_baseline:.4f}")

    if col == 'humidity':
        lag = 4
    if col == 'wind_speed':
        lag = 20
    elif col == 'meanpressure':
        lag = 1
    
    pred_wfv = []
    history = list(y_train)
    for i in range(len(y_test)):
        model = AutoReg(history, lags = 4).fit()
        next_pred = model.forecast()
        pred_wfv.append(next_pred[0])
        history.append(y_test.iloc[i])

    # Convert predictions to Pandas Series (if needed)
    pred_wfv = pd.Series(pred_wfv, index=y_test.index)

    # Re_calculate the MAE again
    test_mae = mean_absolute_error(y_test, pred_wfv)

    if test_mae < mae_baseline:
        print(f"{col} Has beaten the baseline model with a test MAE of {test_mae:.4f}")
    else:
        print(f"{col} Has not beaten the baseline model with a test MAE of {test_mae:.4f}")

    # Plot the results
    df_pred_test = pd.DataFrame({
        'train_df': y_train,
        'test_df': y_test,
        'pred_wfv': pred_wfv
    })
    fig = px.line(df_pred_test, labels = {'value': 'meantemp'})
    fig.show();
    
