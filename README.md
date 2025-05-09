# DNE Commerce Optimizer Pro

A powerful AI-driven analytics platform for e-commerce optimization, providing real-time insights, predictive modeling, and strategic recommendations.

## Features

### 📈 Sales Intelligence
- Real-time sales heatmap visualization
- Product performance analysis with revenue and sentiment metrics
- Dynamic data updates and trend analysis

### 🧑💻 Customer Insights
- Customer value analysis with spending patterns
- Sentiment analysis from customer reviews
- Customer segmentation and behavior tracking

### 📦 Inventory Optimizer
- Real-time inventory status monitoring
- Smart restock recommendations
- Lead time and stockout prediction

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run dne.py
```

## Data Structure

The application uses the following data files (automatically generated if missing):
- `data/transactions.csv`: Customer transaction records
- `data/products.csv`: Product catalog
- `data/inventory.csv`: Current inventory levels
- `data/competitor_prices.csv`: Market price tracking
- `data/users.csv`: Customer information

## Dependencies

- `pandas>=1.5.0`: Data manipulation and analysis
- `numpy>=1.21.0`: Numerical computations
- `streamlit>=1.22.0`: Web application framework
- `plotly>=5.13.0`: Interactive visualizations
- `scikit-learn>=1.0.0`: Machine learning models
- `statsmodels>=0.13.0`: Statistical analysis
- `textblob>=0.17.0`: Sentiment analysis
- `faker>=18.0.0`: Synthetic data generation

## Usage

1. Launch the application using `streamlit run dne.py`
2. The dashboard will automatically load and display:
   - Executive summary with key metrics
   - Sales intelligence visualizations
   - Customer insights and segmentation
   - Inventory optimization recommendations

3. Use the sidebar to:
   - Simulate real-time data updates
   - Refresh analytics
   - Monitor system status

## Development

The application is built with a modular architecture:
- Data generation and loading
- Cached model initialization
- Visualization components
- Dashboard layout and interaction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
