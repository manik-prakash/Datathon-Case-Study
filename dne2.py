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

# ====================== DATA GENERATION ======================
def generate_product_data():
    """Generate product catalog with competitor prices"""
    df = pd.DataFrame({
        'product_id': [f'PROD-{str(i).zfill(5)}' for i in range(1000, 1020)],
        'product_name': [Faker().catch_phrase() for _ in range(20)],
        'category': random.choices(['Electronics', 'Fashion', 'Home', 'Books'], k=20),
        'base_price': np.round(np.random.uniform(10, 500, 20), 2),
        'cost_price': np.round(np.random.uniform(5, 400, 20), 2)
    })
    # Add competitor prices
    df['competitor1_price'] = df['base_price'] * np.random.uniform(0.85, 1.2, size=20)
    df['competitor2_price'] = df['base_price'] * np.random.uniform(0.9, 1.15, size=20)
    df['competitor3_price'] = df['base_price'] * np.random.uniform(0.8, 1.1, size=20)
    return df

def generate_user_data():
    """Generate user base with purchasing behavior"""
    fake = Faker()
    df = pd.DataFrame({
        'user_id': [f'U{1000+i}' for i in range(50)],
        'join_date': [fake.date_between(start_date='-2y') for _ in range(50)],
        'location': [fake.country() for _ in range(50)],
        'loyalty_tier': random.choices(['Basic', 'Silver', 'Gold'], weights=[70, 20, 10], k=50),
        'avg_order_value': np.round(np.random.uniform(50, 500, 50), 2)
    })
    df['last_purchase'] = [fake.date_between(start_date='-30d') for _ in range(50)]
    return df

def generate_transaction_data():
    """Generate transaction history with dynamic pricing"""
    fake = Faker()
    products = generate_product_data()
    users = generate_user_data()
    
    transactions = []
    for _ in range(1000):
        product = products.sample(1).iloc[0]
        user = users.sample(1).iloc[0]
        base_price = product['base_price']
        
        # Dynamic pricing logic
        day_factor = 1 + (datetime.now().weekday()/20)  # Higher prices mid-week
        time_factor = 1 + (datetime.now().hour/50)      # Higher prices during daytime
        price = base_price * np.random.uniform(0.85, 1.15) * day_factor * time_factor
        
        transactions.append({
            'event_id': f'E{10000+_}',
            'user_id': user['user_id'],
            'product_id': product['product_id'],
            'timestamp': fake.date_time_between(start_date='-30d'),
            'quantity': np.random.randint(1, 5),
            'price_paid': np.round(price, 2),
            'event_type': random.choices(['purchase', 'browse', 'cart_add'], weights=[3, 5, 2])[0],
            'review': fake.sentence() if random.random() > 0.7 else None
        })
    return pd.DataFrame(transactions)

# ====================== DATA LOADING ======================
@st.cache_data(ttl=300, show_spinner="Analyzing Data...")
def load_data():
    """Load or generate demo data with sentiment analysis"""
    try:
        transactions = generate_transaction_data()
        products = generate_product_data()
        users = generate_user_data()
        
        # Sentiment analysis
        transactions['sentiment'] = transactions['review'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if x else None)
            
        return {'transactions': transactions, 'products': products, 'users': users}
        
    except Exception as e:
        st.error(f"Data initialization error: {str(e)}")
        st.stop()

# ====================== ANALYTICS COMPONENTS ======================
def create_sales_dashboard(transactions):
    """Interactive sales performance dashboard"""
    # Resample data
    daily_sales = transactions.set_index('timestamp').resample('D').agg({
        'price_paid': 'sum',
        'quantity': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}], [{"colspan": 2}, None]],
                        subplot_titles=("Daily Sales Trend", "Revenue by Category", "Customer Activity"))
    
    # Sales Trend
    fig.add_trace(go.Bar(
        x=daily_sales['timestamp'], y=daily_sales['price_paid'],
        name='Revenue', marker_color='#636EFA'
    ), row=1, col=1)
    
    # Revenue by Category
    category_rev = transactions.merge(st.session_state.data['products'], on='product_id') \
                     .groupby('category')['price_paid'].sum().reset_index()
    fig.add_trace(go.Pie(
        labels=category_rev['category'], values=category_rev['price_paid'],
        hole=0.4, marker_colors=px.colors.qualitative.Pastel
    ), row=1, col=2)
    
    # Customer Activity
    fig.add_trace(go.Scatter(
        x=daily_sales['timestamp'], y=daily_sales['user_id'],
        name='Active Customers', line=dict(color='#FFA15A'),
        fill='tozeroy'
    ), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False, margin=dict(t=40))
    return fig

def create_price_analysis(selected_product):
    """Dynamic pricing analysis with competitor comparison"""
    products = st.session_state.data['products']
    transactions = st.session_state.data['transactions']
    
    product_data = products[products['product_id'] == selected_product].iloc[0]
    transactions = transactions[transactions['product_id'] == selected_product]
    
    # Price History Analysis
    price_history = transactions.set_index('timestamp').resample('D')['price_paid'].mean().reset_index()
    price_history = price_history.rename(columns={'price_paid': 'average_price'})
    
    # Competitive Analysis
    competitor_prices = {
        'Us': product_data['base_price'],
        'Competitor 1': product_data['competitor1_price'],
        'Competitor 2': product_data['competitor2_price'],
        'Competitor 3': product_data['competitor3_price']
    }
    
    # Create visualizations
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                        subplot_titles=("Price Trend & Forecast", "Market Price Comparison"))
    
    # Price Trend
    fig.add_trace(go.Scatter(
        x=price_history['timestamp'], y=price_history['average_price'],
        name='Our Price', line=dict(color='#636EFA')
    ), row=1, col=1)
    
    # Price Forecast
    try:
        model = ARIMA(price_history['average_price'], order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        forecast_dates = pd.date_range(start=price_history['timestamp'].max() + timedelta(days=1), periods=7)
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast,
            name='7-Day Forecast', line=dict(color='#FFA15A', dash='dot')
        ), row=1, col=1)
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
    
    # Competitor Comparison
    fig.add_trace(go.Bar(
        x=list(competitor_prices.keys()), y=list(competitor_prices.values()),
        marker_color=['#2CA02C' if x == 'Us' else '#636EFA' for x in competitor_prices.keys()]
    ), row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False, margin=dict(t=40))
    
    # Profitability Analysis
    margin = ((product_data['base_price'] - product_data['cost_price']) / product_data['base_price']) * 100
    price_recommendation = f"""
    ## 💡 Pricing Insights
    - **Current Margin**: {margin:.1f}% 
    - **Recommended Price**: ${product_data['base_price'] * 0.98:.2f} (-2% for competitiveness)
    - **Minimum Viable Price**: ${product_data['cost_price'] * 1.15:.2f} (15% markup)
    """
    
    return fig, price_recommendation

# ====================== DASHBOARD LAYOUT ======================
def main():
    # Initialize session state
    if 'data' not in st.session_state:
        with st.spinner("🚀 Launching Commerce Optimizer..."):
            st.session_state.data = load_data()
            st.session_state.models = {
                'nmf': NMF(n_components=5, init='nndsvda'),
                'kmeans': KMeans(n_clusters=4, n_init=10)
            }

    # Header Section
    st.title("📊 DNE Commerce Optimizer Pro")
    st.markdown("**AI-Powered Retail Intelligence Platform** | *Real-time Analytics • Predictive Modeling • Dynamic Pricing*")

    # Key Metrics
    st.header("📌 Executive Summary")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Revenue", f"${st.session_state.data['transactions']['price_paid'].sum():,.0f}")
    with cols[1]:
        st.metric("Active Products", st.session_state.data['products']['product_id'].nunique())
    with cols[2]:
        st.metric("Avg. Order Value", f"${st.session_state.data['transactions'].groupby('user_id')['price_paid'].sum().mean():.0f}")
    with cols[3]:
        sentiment = st.session_state.data['transactions']['sentiment'].mean()
        st.metric("Customer Sentiment", f"{sentiment:.2f} ★" if not pd.isna(sentiment) else "N/A")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Sales Dashboard", "🧑💻 Customer Insights", "💰 Price Optimizer"])

    with tab1:
        st.plotly_chart(create_sales_dashboard(st.session_state.data['transactions']), use_container_width=True)

    with tab2:
        st.header("Customer Behavior Analysis")
        # Add customer segmentation components here

    with tab3:
        st.header("Dynamic Price Optimization")
        selected_product = st.selectbox(
            "Select Product for Analysis",
            st.session_state.data['products']['product_id'],
            help="Analyze pricing strategy and market position"
        )
        
        fig, insights = create_price_analysis(selected_product)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(insights)

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")
    if st.sidebar.button("🔄 Simulate Market Changes"):
        new_trans = generate_transaction_data().sample(50)
        st.session_state.data['transactions'] = pd.concat([st.session_state.data['transactions'], new_trans])
        st.experimental_rerun()

    st.sidebar.download_button(
        label="📥 Export Report",
        data=st.session_state.data['transactions'].to_csv().encode('utf-8'),
        file_name='commerce_analytics.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()