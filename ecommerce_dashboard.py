# 📊 E-Commerce Website Traffic Analytics Dashboard using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from datetime import datetime

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="E-commerce Analytics Dashboard", layout="wide")
st.title("📊 E-commerce Website Traffic Analytics Dashboard")
st.markdown("""
This dashboard analyzes website traffic data to understand user behavior, 
optimize user experience, and improve conversions.
""")

# ---------------------- Load and Clean Data ----------------------
@st.cache_data

def load_data():
    df = pd.read_csv("E-commerce Website Traffic Analytics.csv")

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Fill missing values
    df['device'] = df['device'].fillna('Unknown')
    df['browser'] = df['browser'].fillna('Unknown')
    df['location'] = df['location'].fillna('Unknown')
    df['duration_seconds'] = df['duration_seconds'].fillna(0)
    df['transactions'] = df['transactions'].fillna(0)
    df['revenue'] = df['revenue'].fillna(0)
    df['bounce'] = df['bounce'].fillna('no').astype(str).str.lower()
    df['bounce_flag'] = df['bounce'].apply(lambda x: x in ['yes', 'true', '1'])
    df['converted'] = df['transactions'] > 0

    # Parse nested JSON fields
    def parse_json_field(field, key):
        try:
            js = json.loads(field)
            val = js.get(key, 'Unknown')
            return str(val) if val else 'Unknown'
        except:
            return 'Unknown'

    df['utm_source'] = df['source_info'].apply(lambda x: parse_json_field(x, 'utm_source'))
    df['utm_medium'] = df['source_info'].apply(lambda x: parse_json_field(x, 'utm_medium'))
    df['utm_campaign'] = df['source_info'].apply(lambda x: parse_json_field(x, 'utm_campaign'))

    def extract_pages(events):
        try:
            ev = json.loads(events)
            return [e['page'] for e in ev if e.get('event') == 'pageview']
        except:
            return []

    df['pageviews'] = df['events'].apply(lambda x: len(extract_pages(x)))
    df['journey'] = df['events'].apply(lambda x: ' → '.join(extract_pages(x)[:5]))

    df['location'] = df['location'].str.title()
    return df

# Load data
df = load_data()

# ---------------------- Sidebar Filters ----------------------
st.sidebar.header("Filter Options")
devices = st.sidebar.multiselect("Select Device:", options=df['device'].unique(), default=df['device'].unique())
locations = st.sidebar.multiselect("Select Location:", options=df['location'].unique(), default=df['location'].unique())

df_filtered = df[(df['device'].isin(devices)) & (df['location'].isin(locations))]

# ---------------------- KPI Metrics ----------------------
st.markdown("## 🔑 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sessions", len(df_filtered))
col2.metric("Avg Duration (min)", round(df_filtered['duration_seconds'].mean() / 60, 2))
col3.metric("Bounce Rate", f"{round(df_filtered['bounce_flag'].mean()*100, 2)}%")
col4.metric("Conversion Rate", f"{round(df_filtered['converted'].mean()*100, 2)}%")

st.divider()

# ---------------------- Traffic Sources ----------------------
st.markdown("## 🌐 Traffic Sources")
sources = df_filtered['utm_source'].value_counts().reset_index()
sources.columns = ['Source', 'Sessions']
fig1 = px.pie(sources, names='Source', values='Sessions', title="Sessions by Traffic Source")
st.plotly_chart(fig1, use_container_width=True)

# ---------------------- Bounce Rate by Browser ----------------------
st.markdown("## 💻 Bounce Rate by Browser")
browser_bounce = df_filtered.groupby('browser')['bounce_flag'].mean().reset_index()
browser_bounce['bounce_flag'] = browser_bounce['bounce_flag'] * 100
fig2 = px.bar(browser_bounce, x='browser', y='bounce_flag', color='browser', title="Bounce Rate (%) by Browser")
st.plotly_chart(fig2, use_container_width=True)

# ---------------------- Conversion Rate by Device ----------------------
st.markdown("## 📱 Conversion Rate by Device")
device_conversion = df_filtered.groupby('device')['converted'].mean().reset_index()
device_conversion['converted'] = device_conversion['converted'] * 100
fig3 = px.bar(device_conversion, x='device', y='converted', color='device', title="Conversion Rate (%) by Device")
st.plotly_chart(fig3, use_container_width=True)

# ---------------------- Revenue by Location ----------------------
st.markdown("## 💰 Revenue by Location")
revenue_loc = df_filtered.groupby('location')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
fig4 = px.bar(revenue_loc, x='location', y='revenue', color='location', title="Revenue by Location")
st.plotly_chart(fig4, use_container_width=True)

# ---------------------- Top User Journeys ----------------------
st.markdown("## 🧭 Top User Journeys")
journeys = df_filtered['journey'].value_counts().reset_index().head(10)
journeys.columns = ['Journey Path', 'Frequency']
fig5 = px.bar(journeys, y='Journey Path', x='Frequency', orientation='h', title="Top 10 User Journeys")
st.plotly_chart(fig5, use_container_width=True)

# ---------------------- Recommendations ----------------------
st.markdown("## ✅ Strategic Recommendations")
st.markdown("""
- 🔍 **Reduce Bounce Rate**: Optimize user experience for high-bounce browsers.
- 📱 **Device Optimization**: Enhance mobile UX if mobile has strong conversions.
- 🎯 **Campaign Focus**: Invest more in top-performing traffic sources.
- 🌎 **Location Targeting**: Target regions generating the most revenue.
- 🧭 **Navigation Improvements**: Simplify paths in common user journeys.
""")
