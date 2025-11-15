"""
Demand Analysis Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.demand import DemandModeler

st.set_page_config(page_title="Demand Analysis", page_icon="📊", layout="wide")

st.title("📊 Demand Intelligence Dashboard")

if 'df' not in st.session_state:
    st.warning("⚠️ Please load data from the main page first")
    st.stop()

df = st.session_state.df

# Initialize modeler
modeler = DemandModeler(df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Demand Overview", "🔥 High Demand Products", "📍 Category Demand", "🎯 Opportunities"])

with tab1:
    st.header("Demand Overview & Segmentation")
    
    # Calculate metrics
    df_segmented = modeler.segment_by_demand()
    summary = modeler.get_demand_summary()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{summary.get('total_reviews', 0):,}")
    
    with col2:
        st.metric("Avg Reviews/Product", f"{summary.get('avg_reviews_per_product', 0):.1f}")
    
    with col3:
        st.metric("Avg Rating", f"{summary.get('avg_rating', 0):.2f}⭐")
    
    with col4:
        pct_reviews = summary.get('pct_with_reviews', 0)
        st.metric("Products with Reviews", f"{pct_reviews:.1f}%")
    
    st.divider()
    
    # Demand segmentation chart
    st.subheader("Demand Segmentation")
    
    if 'demand_segment' in df_segmented.columns:
        segment_counts = df_segmented['demand_segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Product Distribution by Demand Level",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment statistics
        st.subheader("Segment Statistics")
        segment_stats = df_segmented.groupby('demand_segment').agg({
            'discount_price': 'mean',
            'ratings': 'mean',
            'no_of_ratings': ['mean', 'sum'],
            'name': 'count'
        }).round(2)
        st.dataframe(segment_stats, use_container_width=True)

with tab2:
    st.header("🔥 High Demand Products")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        percentile = st.slider("Demand Percentile", 50, 99, 90)
        min_rating = st.slider("Min Rating", 1.0, 5.0, 4.0, 0.5)
    
    with st.spinner("Identifying high-demand products..."):
        high_demand = modeler.identify_high_demand_products(
            percentile=percentile,
            min_rating=min_rating
        )
    
    if not high_demand.empty:
        st.success(f"Found {len(high_demand)} high-demand products")
        
        # Top products chart
        top_products = high_demand.head(20)
        
        fig = px.bar(
            top_products,
            x='no_of_ratings',
            y='name',
            orientation='h',
            color='ratings',
            title=f"Top 20 High-Demand Products (>{percentile}th percentile)",
            labels={'no_of_ratings': 'Number of Reviews', 'name': 'Product'},
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("High Demand Products Detail")
        st.dataframe(high_demand, use_container_width=True, height=400)
        
        # Download
        csv = high_demand.to_csv(index=False)
        st.download_button(
            "📥 Download High Demand Products",
            csv,
            "high_demand_products.csv",
            "text/csv"
        )
    else:
        st.info("No products match the criteria")

with tab3:
    st.header("Category-Level Demand Analysis")
    
    with st.spinner("Analyzing category demand..."):
        category_demand = modeler.analyze_demand_by_category()
    
    if not category_demand.empty:
        # Key insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_demand = category_demand.iloc[0]
            st.metric(
                "Highest Total Demand",
                top_demand['category'],
                f"{top_demand['total_reviews']:,.0f} reviews"
            )
        
        with col2:
            top_intensity = category_demand.nlargest(1, 'demand_intensity').iloc[0]
            st.metric(
                "Highest Demand Intensity",
                top_intensity['category'],
                f"{top_intensity['demand_intensity']:.1f}"
            )
        
        with col3:
            top_share = category_demand.iloc[0]
            st.metric(
                "Largest Market Share",
                top_share['category'],
                f"{top_share['market_share_proxy']:.1f}%"
            )
        
        st.divider()
        
        # Visualizations
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Total Demand by Category")
            
            top_15 = category_demand.head(15)
            fig = px.bar(
                top_15,
                x='category',
                y='total_reviews',
                title="Top 15 Categories by Total Reviews",
                labels={'total_reviews': 'Total Reviews', 'category': 'Category'}
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("Demand Intensity")
            
            fig = px.scatter(
                category_demand.head(20),
                x='product_count',
                y='demand_intensity',
                size='total_reviews',
                color='avg_rating',
                hover_name='category',
                title="Demand Intensity vs Competition",
                labels={
                    'product_count': 'Number of Products',
                    'demand_intensity': 'Demand Intensity',
                    'avg_rating': 'Avg Rating'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Market share
        st.subheader("Market Share Distribution")
        
        top_10_share = category_demand.head(10)
        fig = px.pie(
            top_10_share,
            values='market_share_proxy',
            names='category',
            title="Market Share (Top 10 Categories by Review Volume)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Full data
        st.subheader("Complete Category Analysis")
        st.dataframe(category_demand, use_container_width=True, height=400)

with tab4:
    st.header("Low Competition Opportunities")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        demand_thresh = st.slider("Demand Threshold", 0.1, 1.0, 0.3, 0.1)
        quality_thresh = st.slider("Quality Threshold", 3.0, 5.0, 4.0, 0.1)
    
    with st.spinner("Finding opportunities..."):
        opportunities = modeler.find_low_competition_opportunities(
            demand_threshold=demand_thresh,
            quality_threshold=quality_thresh
        )
    
    if not opportunities.empty:
        st.success(f"💡 Found {len(opportunities)} opportunity categories")
        
        # Opportunity matrix
        fig = px.scatter(
            opportunities,
            x='product_count',
            y='demand_intensity',
            size='total_reviews',
            color='avg_rating',
            hover_name='category',
            title="Opportunity Matrix: Low Competition, Good Demand",
            labels={
                'product_count': 'Competition (# Products)',
                'demand_intensity': 'Demand Intensity',
                'avg_rating': 'Avg Rating'
            },
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data
        st.dataframe(opportunities, use_container_width=True)
        
        # Export
        csv = opportunities.to_csv(index=False)
        st.download_button(
            "📥 Download Opportunities",
            csv,
            "low_competition_opportunities.csv",
            "text/csv"
        )
    else:
        st.info("No categories meet the opportunity criteria. Try adjusting the thresholds.")
    
    st.divider()
    
    # Price-Demand correlation
    st.subheader("Price-Demand Relationship")
    
    with st.spinner("Calculating correlations..."):
        corr_results = modeler.calculate_demand_price_correlation()
    
    if 'overall' in corr_results:
        overall = corr_results['overall']
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Overall Correlation", f"{overall['correlation']:.3f}")
        
        with col_b:
            st.metric("Interpretation", overall['interpretation'].title())
        
        with col_c:
            st.metric("Strength", overall['strength'].title())
        
        if 'by_category' in corr_results:
            st.subheader("Correlation by Category")
            
            corr_df = corr_results['by_category']
            
            fig = px.bar(
                corr_df.head(20),
                x='category',
                y='correlation',
                title="Price-Demand Correlation (Top 20 Categories)",
                labels={'correlation': 'Correlation Coefficient', 'category': 'Category'}
            )
            fig.update_xaxes(tickangle=-45)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
