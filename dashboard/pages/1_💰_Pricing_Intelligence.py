"""
Pricing Intelligence Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.pricing import PricingAnalyzer

st.set_page_config(page_title="Pricing Intelligence", page_icon="💰", layout="wide")

st.title("💰 Pricing Intelligence Dashboard")

# Check if data is loaded
if 'df' not in st.session_state:
    st.warning("⚠️ Please load data from the main page first")
    st.stop()

df = st.session_state.df

# Reset index to avoid any alignment issues
df = df.reset_index(drop=True)

# Initialize analyzer
analyzer = PricingAnalyzer(df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Category Analysis", "🏷️ Brand Analysis", "💸 Discount Strategies", "⚡ Outliers"])

with tab1:
    st.header("Category-Level Pricing")
    
    # Analyze by category
    with st.spinner("Analyzing categories..."):
        category_stats = analyzer.analyze_by_category()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        highest_avg = category_stats.iloc[0]
        st.metric(
            "Highest Avg Price Category",
            highest_avg['category'],
            f"₹{highest_avg['avg_price']:,.0f}"
        )
    
    with col2:
        most_products = category_stats.iloc[0]
        st.metric(
            "Most Products",
            most_products['category'],
            f"{most_products['product_count']:,} products"
        )
    
    with col3:
        highest_competitive = category_stats.nlargest(1, 'competitiveness_index').iloc[0]
        st.metric(
            "Most Competitive",
            highest_competitive['category'],
            f"Index: {highest_competitive['competitiveness_index']:.2f}"
        )
    
    st.divider()
    
    # Interactive chart
    st.subheader("Price Distribution by Category")
    
    top_n = st.slider("Number of categories to display", 5, 30, 15)
    metric = st.selectbox(
        "Sort by",
        ["product_count", "avg_price", "avg_discount", "competitiveness_index"]
    )
    
    top_categories = category_stats.nlargest(top_n, metric)
    
    fig = px.bar(
        top_categories,
        x='category',
        y='avg_price',
        color='avg_discount',
        title=f"Top {top_n} Categories by {metric}",
        labels={'avg_price': 'Average Price (₹)', 'category': 'Category', 'avg_discount': 'Avg Discount %'},
        color_continuous_scale='RdYlGn'
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Detailed Category Statistics")
    st.dataframe(category_stats, use_container_width=True, height=400)

with tab2:
    st.header("Brand-Level Pricing Analysis")
    
    min_products = st.slider("Minimum products per brand", 5, 100, 20)
    
    with st.spinner("Analyzing brands..."):
        brand_stats = analyzer.analyze_by_brand(min_products=min_products)
    
    if not brand_stats.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Premium Brands")
            premium_brands = brand_stats.nlargest(10, 'price_premium_vs_market')
            
            fig = px.bar(
                premium_brands,
                x='brand',
                y='price_premium_vs_market',
                color='avg_rating',
                title="Brand Premium vs Market Average",
                labels={'price_premium_vs_market': 'Premium %', 'brand': 'Brand'},
                color_continuous_scale='Viridis'
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Value Brands")
            value_brands = brand_stats.nsmallest(10, 'price_premium_vs_market')
            
            fig = px.bar(
                value_brands,
                x='brand',
                y='price_premium_vs_market',
                color='avg_rating',
                title="Brand Discount vs Market Average",
                labels={'price_premium_vs_market': 'Premium %', 'brand': 'Brand'},
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("All Brands Statistics")
        st.dataframe(brand_stats, use_container_width=True, height=400)
    else:
        st.info("No brands meet the minimum product threshold")

with tab3:
    st.header("Discount Strategy Analysis")
    
    with st.spinner("Analyzing discount patterns..."):
        discount_stats = analyzer.compare_discount_strategies(top_n=20)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Discount by Category")
        
        fig = px.bar(
            discount_stats.head(15),
            x='main_category',
            y='discount_percent_mean',
            error_y='discount_percent_std',
            title="Discount Percentage (with variability)",
            labels={'discount_percent_mean': 'Avg Discount %', 'main_category': 'Category'}
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Discount Aggressiveness")
        
        fig = px.scatter(
            discount_stats,
            x='discount_percent_mean',
            y='discount_percent_std',
            size='actual_price_mean',
            color='discount_aggressiveness',
            hover_name='main_category',
            title="Discount Strategy Positioning",
            labels={
                'discount_percent_mean': 'Avg Discount %',
                'discount_percent_std': 'Discount Variability',
                'discount_aggressiveness': 'Aggressiveness'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(discount_stats, use_container_width=True)

with tab4:
    st.header("Pricing Outliers")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        method = st.radio("Detection Method", ["IQR", "Z-Score"])
        factor = st.slider("Sensitivity", 1.0, 5.0, 3.0, 0.5)
    
    with col2:
        with st.spinner("Detecting outliers..."):
            outliers = analyzer.find_pricing_outliers(
                method=method.lower().replace('-', ''),
                factor=factor
            )
        
        st.info(f"Found {len(outliers)} pricing outliers")
        
        if not outliers.empty:
            # Split by type
            overpriced = outliers[outliers['outlier_type'] == 'Overpriced']
            underpriced = outliers[outliers['outlier_type'] == 'Underpriced']
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader(f"🔴 Overpriced ({len(overpriced)})")
                st.dataframe(overpriced.head(20), use_container_width=True)
            
            with col_b:
                st.subheader(f"🟢 Underpriced ({len(underpriced)})")
                st.dataframe(underpriced.head(20), use_container_width=True)
            
            # Distribution
            st.subheader("Outlier Distribution")
            
            # Clean data for plotting - remove NaN values and ensure positive sizes
            outliers_clean = outliers.copy()
            outliers_clean['no_of_ratings'] = outliers_clean['no_of_ratings'].fillna(1)
            outliers_clean = outliers_clean[outliers_clean['no_of_ratings'] > 0]
            
            fig = px.scatter(
                outliers_clean,
                x='discount_price',
                y='ratings',
                color='outlier_type',
                size='no_of_ratings',
                hover_name='name',
                title="Pricing Outliers: Price vs Rating",
                labels={'discount_price': 'Price (₹)', 'ratings': 'Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)
