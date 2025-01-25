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

# ====================== FILE PATH CONFIGURATION ======================
def get_file_paths():
    """File path configuration section in sidebar"""
    st.sidebar.header("📂 Data Source Configuration")
    return {
        'transactions': st.sidebar.text_input("Transactions CSV Path", "data/transactions.csv"),
        'products': st.sidebar.text_input("Products CSV Path", "data/products.csv"),
        'users': st.sidebar.text_input("Users CSV Path", "data/users.csv")
    }

# ====================== DATA LOADING ======================
@st.cache_data(ttl=300, show_spinner="Analyzing Data...")
def load_data(file_paths):
    """Load data from configured file paths with fallback to generated data"""
    dfs = {}
    
    try:
        # Load core datasets
        for name in ['transactions', 'products', 'users']:
            if os.path.exists(file_paths[name]):
                dfs[name] = pd.read_csv(
                    file_paths[name],
                    parse_dates=['timestamp'] if name == 'transactions' else None
                )
        
        # Generate missing data
        if 'transactions' not in dfs:
            dfs['transactions'] = generate_transaction_data()
        
        if 'products' not in dfs:
            dfs['products'] = generate_product_data()
        
        if 'users' not in dfs:
            dfs['users'] = generate_user_data()
        
        # Generate sentiment analysis
        dfs['transactions']['sentiment'] = dfs['transactions']['review'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if x else None)
            
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.stop()
        
    return dfs

# ====================== DATA GENERATION ======================
def generate_product_data():
    """Generate product catalog"""
    return pd.DataFrame({
        'product_id': [f'PROD-{str(i).zfill(5)}' for i in range(1000, 1020)],
        'product_name': [Faker().catch_phrase() for _ in range(20)],
        'category': random.choices(['Electronics', 'Fashion', 'Home', 'Books'], k=20),
        'base_price': np.round(np.random.uniform(10, 500, 20), 2)
    })

def generate_user_data():
    """Generate user base"""
    fake = Faker()
    return pd.DataFrame({
        'user_id': [f'U{1000+i}' for i in range(50)],
        'join_date': [fake.date_between(start_date='-2y') for _ in range(50)],
        'location': [fake.country() for _ in range(50)],
        'loyalty_tier': random.choices(['Basic', 'Silver', 'Gold'], weights=[70, 20, 10], k=50)
    })

def generate_transaction_data():
    """Generate transaction history"""
    fake = Faker()
    products = generate_product_data()
    users = generate_user_data()
    
    transactions = []
    for _ in range(1000):
        product = random.choice(products['product_id'])
        base_price = products.loc[products['product_id'] == product, 'base_price'].values[0]
        transactions.append({
            'event_id': f'E{10000+_}',
            'user_id': random.choice(users['user_id']),
            'product_id': product,
            'timestamp': fake.date_time_between(start_date='-30d'),
            'quantity': np.random.randint(1, 5),
            'price_paid': np.round(base_price * np.random.uniform(0.85, 1.15), 2),
            'event_type': random.choices(['purchase', 'browse', 'cart_add'], weights=[3, 5, 2])[0],
            'review': fake.sentence() if random.random() > 0.7 else None
        })
    return pd.DataFrame(transactions)

# ====================== VISUALIZATION COMPONENTS ======================
def create_sales_heatmap(df):
    """Create weekly sales heatmap"""
    df = df.set_index('timestamp').resample('D')['price_paid'].sum().reset_index()
    df['day'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.isocalendar().week
    return px.density_heatmap(
        df, x='day', y='week', z='price_paid',
        title="Weekly Sales Heatmap", color_continuous_scale='Viridis'
    )

def create_product_performance_chart(products, trans):
    """Create product performance matrix"""
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
        hovermode="x unified",
        width=1200  # Adjust the width as needed
    )
    return fig

def create_price_analysis(selected_product, transactions, products):
    """Create price history and recommendation components"""
    product_trans = transactions[transactions['product_id'] == selected_product]
    base_price = products[products['product_id'] == selected_product]['base_price'].values[0]
    
    # Price History Analysis
    price_history = product_trans.set_index('timestamp').resample('D')['price_paid'].mean().reset_index()
    price_history = price_history.rename(columns={'price_paid': 'average_price'})
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                        subplot_titles=("Price Evolution", "Weekly Price Patterns"))
    
    # Price Evolution
    fig.add_trace(go.Scatter(
        x=price_history['timestamp'], y=price_history['average_price'],
        name='Actual Price', line=dict(color='#636EFA'),
        hovertemplate="%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)
    
    # Base price reference
    fig.add_hline(y=base_price, line_dash="dot", 
                 annotation_text="Base Price", 
                 line_color="#2CA02C", row=1, col=1)
    
    # Weekly Patterns
    price_history['day_of_week'] = price_history['timestamp'].dt.day_name()
    weekly_avg = price_history.groupby('day_of_week')['average_price'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]).reset_index()
    
    fig.add_trace(go.Bar(
        x=weekly_avg['day_of_week'], y=weekly_avg['average_price'],
        marker_color='#FFA15A', name='Average Daily Price',
        hovertemplate="%{x}<br>Avg: $%{y:.2f}<extra></extra>"
    ), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False, margin=dict(t=40))
    
    # Recommendation Logic
    best_day = weekly_avg.loc[weekly_avg['average_price'].idxmin(), 'day_of_week']
    best_day_avg = weekly_avg['average_price'].min()
    price_diff = ((base_price - best_day_avg)/base_price)*100
    
    recommendation = f"""
    🎯 **Optimal Purchase Strategy**  
    - **Best Day to Buy**: {best_day} (Average ${best_day_avg:.2f})  
    - **Discount vs Base**: {price_diff:.1f}% lower than base price  
    - **Historical Low**: ${price_history['average_price'].min():.2f}  
    - **Current Price**: ${price_history['average_price'].iloc[-1]:.2f}
    """
    
    return fig, recommendation

def predict_future_price(selected_product, transactions):
    """Predict future price using ARIMA model"""
    product_trans = transactions[transactions['product_id'] == selected_product]
    price_history = product_trans.set_index('timestamp').resample('D')['price_paid'].mean().dropna()
    
    model = ARIMA(price_history, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_history.index, y=price_history,
        mode='lines', name='Historical Prices'
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast,
        mode='lines', name='Predicted Prices'
    ))
    
    fig.update_layout(
        title="Future Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    
    return fig

def should_wait_for_new_product(selected_product, products):
    """Randomly decide if a new product from the same brand is coming soon"""
    brand = products[products['product_id'] == selected_product]['product_name'].values[0].split()[0]
    new_product_coming = random.choice([True, False])
    
    if new_product_coming:
        message = f"A new product from {brand} is expected to be released soon. You might want to wait!"
    else:
        message = f"No new products from {brand} are expected soon. It's a good time to buy now!"
    
    return message

# ====================== DASHBOARD LAYOUT ======================
def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.file_paths = get_file_paths()
        with st.spinner("🚀 Launching DNE Commerce Optimizer Pro..."):
            st.session_state.models = {
                'nmf': NMF(n_components=5, init='nndsvda'),
                'kmeans': KMeans(n_clusters=4, n_init=10)
            }
            st.session_state.data = load_data(st.session_state.file_paths)

    # Header Section
    st.title("📊 DNE Commerce Optimizer Pro")
    st.markdown("**AI-Powered Insights Platform** | *Real-time Analytics • Predictive Modeling • Strategic Recommendations*")

    # Executive Summary
    st.header("📌 Executive Summary")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Revenue", f"${st.session_state.data['transactions']['price_paid'].sum():,.2f}")
    with cols[1]:
        st.metric("Active Customers", st.session_state.data['transactions']['user_id'].nunique())
    with cols[2]:
        avg_sentiment = st.session_state.data['transactions']['sentiment'].mean()
        st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}" if not pd.isna(avg_sentiment) else "N/A")

    # Main Tabs
    tab1, tab2 = st.tabs(["📈 Sales Intelligence", "🧑💻 Customer Insights"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sales_heatmap(st.session_state.data['transactions']), use_container_width=True)
        with col2:
            st.plotly_chart(create_product_performance_chart(
                st.session_state.data['products'], 
                st.session_state.data['transactions']
            ), use_container_width=True)

    with tab2:
        st.header("🧑💻 Customer Insights")
        
        # Product Selector
        products = st.session_state.data['products']
        selected_product = st.selectbox(
            "🔍 Select Product for Analysis",
            products['product_id'],
            help="Analyze pricing history and optimal purchase timing"
        )
        
        # Product Details
        product_name = products[products['product_id'] == selected_product]['product_name'].values[0]
        st.subheader(f"📦 {product_name} Analysis")
        
        # Generate Analysis
        price_fig, recommendation = create_price_analysis(
            selected_product,
            st.session_state.data['transactions'],
            st.session_state.data['products']
        )
        
        # Layout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(price_fig, use_container_width=True)
        with col2:
            st.markdown("### 💡 Purchase Recommendations")
            st.markdown(recommendation)
        
        # Future Price Prediction
        st.subheader("📈 Future Price Prediction")
        future_price_fig = predict_future_price(selected_product, st.session_state.data['transactions'])
        st.plotly_chart(future_price_fig, use_container_width=True)
        
        # Should Wait for New Product
        st.subheader("⏳ Should You Wait for a New Product?")
        wait_message = should_wait_for_new_product(selected_product, st.session_state.data['products'])
        st.markdown(wait_message)

    # Sidebar Controls
    st.sidebar.header("⚡ Live Updates")
    if st.sidebar.button("🔄 Simulate Real-time Data"):
        new_trans = generate_transaction_data().sample(50)
        st.session_state.data['transactions'] = pd.concat([st.session_state.data['transactions'], new_trans])
        st.experimental_rerun()

    if st.sidebar.button("🗂️ Reload Data Sources"):
        st.session_state.data = load_data(st.session_state.file_paths)
        st.experimental_rerun()

if __name__ == "__main__":
    main()