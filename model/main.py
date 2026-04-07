import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score


import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

def get_clean_data():
    df = pd.read_csv('data/train.csv')
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    df['ship_date'] = pd.to_datetime(df['ship_date'], dayfirst=True)
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    df['dayofweek'] = df['order_date'].dt.dayofweek
    df['quarter'] = df['order_date'].dt.quarter
    df['shipping_days'] = (df['ship_date'] - df['order_date']).dt.days
    drop_cols =[
        'row_id', 'order_id', 'customer_id',
        'customer_name', 'product_id',
        'ship_date', 'ship_mode', 
        'postal_code', 'region'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')
    