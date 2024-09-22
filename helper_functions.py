import subprocess
import sys

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