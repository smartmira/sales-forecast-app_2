import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import create_features
import plotly.express as px


st.set_page_config(page_title="Sales Forecast", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

FEATURE_COLS = [
    'sub-category', 'state',
    'lag_1', 'lag_2', 'rolling_3',
    'time_index', 'month_sin', 'month_cos',
    'momentum_1', 'subcategory_avg'
]


df = pd.read_csv("train.csv")

# Clean columns
df.columns = df.columns.str.lower().str.strip()
df.columns = df.columns.str.replace(" ", "_")

# Convert dates
df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)

st.title("📊 Sales Forecasting Dashboard")

st.sidebar.header("Controls")

state = st.sidebar.selectbox("Select State", df['state'].unique())
category = st.sidebar.selectbox("Select Category", df['sub-category'].unique())

filtered_df = df[
    (df['state'] == state) &
    (df['sub-category'] == category)
].copy()

filtered_df = filtered_df.sort_values('order_date')

fig = px.line(
    filtered_df,
    x='order_date',
    y='sales',
    title="📈 Historical Sales"
)

st.plotly_chart(fig, use_container_width=True)

future_dates = pd.date_range(
    start=filtered_df['order_date'].max(),
    end="2027-12-31",
    freq='M'
)

future_df = pd.DataFrame({
    'order_date': future_dates,
    'state': state,
    'sub-category': category,
    'sales': np.nan   # VERY IMPORTANT
})

full_df = pd.concat([filtered_df, future_df]).reset_index(drop=True)

for i in range(len(full_df)):

    # Recompute features every step
    full_df = create_features(full_df)

    # If future row (no sales yet)
    if pd.isna(full_df.loc[i, 'sales']):

        row = full_df.loc[i, FEATURE_COLS]

        row_df = pd.DataFrame([row])

        pred = model.predict(row_df)[0]

        # Prevent negative predictions
        pred = max(pred, 0)

        full_df.loc[i, 'sales'] = pred

future_only = full_df[
    full_df['order_date'] > filtered_df['order_date'].max()
]

fig2 = px.line(
    future_only,
    x='order_date',
    y='sales',
    title="🔮 Sales Forecast (Up to 2027)"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("📥 Download Forecast")

csv = future_only.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="sales_forecast_2027.csv",
    mime="text/csv"
)

st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)