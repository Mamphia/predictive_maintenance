def preprocess_input(df):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # Example preprocessing logic extracted from notebook
    df['cycle'] = df.groupby('unit_number').cumcount() + 1
    # More feature engineering and rolling windows...
    # df['sensor_1_roll_mean'] = df['sensor_1'].rolling(window=5).mean()
    
    # Fill or drop NaNs, normalize
    df = df.fillna(method='bfill')
    scaler = MinMaxScaler()
    scaled_cols = df.columns[df.dtypes != 'object']
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])
    
    return df
