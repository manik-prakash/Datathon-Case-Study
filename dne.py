import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from datetime import datetime, timedelta
from faker import Faker
import random
import warnings
import os

# Configuration
warnings.filterwarnings("ignore")
st.set_page_config(page_title="DNE Commerce Optimizer Pro", layout="wide", page_icon="📈")
pd.options.plotting.backend = "plotly"

# ====================== DATA GENERATION (REPLACE WITH REAL DATA) ======================
def generate_sample_data():
    """Generate synthetic data if real files are missing"""
    fake = Faker()
    products = pd.DataFrame({
        'product_id': [f'P{1000+i}' for i in range(20)],
        'product_name': [fake.catch_phrase() for _ in range(20)],
        'category': random.choices(['Electronics', 'Fashion', 'Home', 'Books'], k=20),
        'base_price': np.random.uniform(10, 500, 20)
    })
    
    users = pd.DataFrame({
        'user_id': [f'U{1000+i}' for i in range(50)],
        'join_date': [fake.date_between(start_date='-2y') for _ in range(50)],
        'location': [fake.country() for _ in range(50)]
    })
    
    transactions = pd.DataFrame({
        'event_id': [f'E{10000+i}' for i in range(1000)],
        'user_id': random.choices(users['user_id'], k=1000),
        'product_id': random.choices(products['product_id'], k=1000),
        'timestamp': [fake.date_time_between(start_date='-30d') for _ in range(1000)],
        'quantity': np.random.randint(1, 5, 1000),
        'price_paid': np.random.uniform(10, 500, 1000),
        'event_type': random.choices(['purchase', 'browse', 'cart_add'], weights=[3, 5, 2], k=1000),
        'review': [fake.sentence() if random.random() > 0.7 else None for _ in range(1000)]
    })
    
    inventory = pd.DataFrame({
        'product_id': products['product_id'],
        'current_stock': np.random.randint(0, 100, 20),
        'restock_level': np.random.randint(10, 30, 20),
        'lead_time': np.random.randint(1, 14, 20)
    })
    
    competitors = pd.DataFrame({
        'product_id': random.choices(products['product_id'], k=100),
        'price': np.random.uniform(5, 600, 100),
        'timestamp': [fake.date_time_between(start_date='-7d') for _ in range(100)]
    })
    
    return {
        'trans': transactions,
        'products': products,
        'inventory': inventory,
        'competitors': competitors,
        'users': users
    }

# ====================== CACHED FUNCTIONS ======================
@st.cache_resource(show_spinner="Initializing AI Models...")
def load_models():
    return {
        'nmf': NMF(n_components=5, init='nndsvda'),
        'kmeans': KMeans(n_clusters=4, n_init=10),
        'arima': lambda x: ARIMA(x, order=(1,1,1)).fit()
    }

@st.cache_data(ttl=300, show_spinner="Analyzing Data...")
def load_data():
    if not os.path.exists('data'):
        os.makedirs('data')
        sample_data = generate_sample_data()
        for name, df in sample_data.items():
            df.to_csv(f'data/{name}.csv', index=False)
    
    dfs = {}
    try:
        dfs['trans'] = pd.read_csv('data/trans.csv', parse_dates=['timestamp'])
        dfs['products'] = pd.read_csv('data/products.csv')
        dfs['inventory'] = pd.read_csv('data/inventory.csv')
        dfs['competitors'] = pd.read_csv('data/competitors.csv')
        dfs['users'] = pd.read_csv('data/users.csv')
        
        # Handle missing review column
        if 'review' not in dfs['trans']:
            dfs['trans']['review'] = None
            
        dfs['trans']['sentiment'] = dfs['trans']['review'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if x else None)
            
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.stop()
        
    return dfs

# ====================== VISUALIZATION COMPONENTS ======================
def create_sales_heatmap(df):
    df = df.set_index('timestamp').resample('D')['price_paid'].sum().reset_index()
    df['day'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.isocalendar().week
    return px.density_heatmap(
        df, x='day', y='week', z='price_paid',
        title="Weekly Sales Heatmap", color_continuous_scale='Viridis'
    )

def create_product_performance_chart(products, trans):
    merged = trans.merge(products, on='product_id', how='left')
    performance = merged.groupby('product_id').agg({
        'price_paid': 'sum',
        'quantity': 'sum',
        'sentiment': 'mean'
    }).reset_index().fillna(0)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=performance['product_id'], y=performance['price_paid'],
        name='Revenue', marker_color='#636EFA'
    ))
    fig.add_trace(go.Scatter(
        x=performance['product_id'], y=performance['sentiment'],
        name='Sentiment', line=dict(color='#EF553B')), secondary_y=True)
    
    fig.update_layout(
        title="Product Performance Matrix",
        xaxis_title="Product ID",
        yaxis_title="Revenue",
        yaxis2_title="Sentiment Score",
        hovermode="x unified"
    )
    return fig

# ====================== DASHBOARD LAYOUT ======================
def main():
    if 'data' not in st.session_state:
        with st.spinner("🚀 Launching DNE Commerce Optimizer Pro..."):
            st.session_state.models = load_models()
            st.session_state.data = load_data()

    # Header Section
    st.title("📊 DNE Commerce Optimizer Pro")
    st.markdown("**AI-Powered Insights Platform** | *Real-time Analytics • Predictive Modeling • Strategic Recommendations*")

    # Executive Summary
    st.header("📌 Executive Summary")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Revenue", f"${st.session_state.data['trans']['price_paid'].sum():,.2f}")
    with cols[1]:
        st.metric("Active Customers", st.session_state.data['trans']['user_id'].nunique())
    with cols[2]:
        st.metric("Inventory Turnover", 
                 f"{st.session_state.data['trans'].shape[0]/st.session_state.data['inventory']['current_stock'].sum():.1f}x")
    with cols[3]:
        avg_sentiment = st.session_state.data['trans']['sentiment'].mean()
        st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}" if not pd.isna(avg_sentiment) else "N/A")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Sales Intelligence", "🧑💻 Customer Insights", "📦 Inventory Optimizer"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sales_heatmap(st.session_state.data['trans']), use_container_width=True)
        with col2:
            st.plotly_chart(create_product_performance_chart(
                st.session_state.data['products'], 
                st.session_state.data['trans']
            ), use_container_width=True)

    # Sidebar Updates
    st.sidebar.header("⚡ Live Updates")
    if st.sidebar.button("🔄 Simulate Real-time Data"):
        new_trans = generate_sample_data()['trans'].sample(50)
        st.session_state.data['trans'] = pd.concat([st.session_state.data['trans'], new_trans])
        st.experimental_rerun()

if __name__ == "__main__":
    main()