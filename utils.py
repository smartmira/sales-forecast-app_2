import numpy as np
import pandas as pd

def create_features(df):
    df = df.sort_values(['state', 'sub-category', 'order_date'])

    df['lag_1'] = df.groupby(['state', 'sub-category'])['sales'].shift(1)
    df['lag_2'] = df.groupby(['state', 'sub-category'])['sales'].shift(2)

    df['rolling_3'] = df.groupby(['state', 'sub-category'])['sales'] \
        .transform(lambda x: x.rolling(3).mean())

    df['time_index'] = (
        (df['order_date'] - df['order_date'].min()).dt.days
    )

    df['month'] = df['order_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['momentum_1'] = df['lag_1'] - df['lag_2']

    df['subcategory_avg'] = (
        df.groupby(['sub-category', 'order_date'])['sales']
        .transform('mean')
    )

    return df