"""
Opportunity Detection Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.opportunities import OpportunityDetector

st.set_page_config(page_title="Market Opportunities", page_icon="🎯", layout="wide")

st.title("🎯 Market Opportunity Dashboard")

if 'df' not in st.session_state:
    st.warning("⚠️ Please load data from the main page first")
    st.stop()

df = st.session_state.df

# Initialize detector
detector = OpportunityDetector(df)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Market Gaps", 
    "⚔️ Price Wars", 
    "💎 Premium Segments", 
    "💰 Value Plays",
    "📊 Opportunity Matrix"
])

with tab1:
    st.header("Market Gap Analysis")
    st.info("Find categories with high demand but low competition")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        min_demand_pct = st.slider("Min Demand Percentile", 50, 95, 60)
        max_comp_pct = st.slider("Max Competition Percentile", 10, 50, 40)
    
    with st.spinner("Analyzing market gaps..."):
        gaps = detector.find_market_gaps(
            min_demand_percentile=min_demand_pct,
            max_competition_percentile=max_comp_pct
        )
    
    if not gaps.empty:
        st.success(f"🎯 Found {len(gaps)} market gap opportunities!")
        
        # Top opportunities
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            top_opp = gaps.iloc[0]
            st.metric(
                "Best Opportunity",
                top_opp['category'],
                f"Score: {top_opp['opportunity_score']:.1f}"
            )
        
        with col_b:
            st.metric(
                "Avg Demand Intensity",
                f"{gaps['demand_intensity'].mean():,.0f}",
                "reviews/product"
            )
        
        with col_c:
            st.metric(
                "Avg Competition",
                f"{gaps['product_count'].mean():,.0f}",
                "products"
            )
        
        st.divider()
        
        # Visualization
        fig = px.scatter(
            gaps,
            x='product_count',
            y='total_demand',
            size='opportunity_score',
            color='avg_rating',
            hover_name='category',
            title="Market Gaps: High Demand + Low Competition",
            labels={
                'product_count': 'Competition (# Products)',
                'total_demand': 'Total Demand (Reviews)',
                'opportunity_score': 'Opportunity Score',
                'avg_rating': 'Avg Rating'
            },
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(gaps, use_container_width=True)
        
        # Download
        csv = gaps.to_csv(index=False)
        st.download_button("📥 Download Market Gaps", csv, "market_gaps.csv", "text/csv")
    else:
        st.warning("No market gaps found with current criteria. Try adjusting the thresholds.")

with tab2:
    st.header("Price War Detection")
    st.info("Identify products in intense competitive pricing battles")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        discount_thresh = st.slider("Min Discount %", 10, 50, 30)
        demand_thresh = st.slider("Demand Percentile", 50, 95, 70)
    
    with st.spinner("Detecting price wars..."):
        price_wars = detector.detect_price_wars(
            discount_threshold=discount_thresh,
            demand_threshold=demand_thresh
        )
    
    if not price_wars.empty:
        st.warning(f"⚔️ {len(price_wars)} products in price wars detected!")
        
        # Metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Avg Discount", f"{price_wars['discount_percent'].mean():.1f}%")
        
        with col_b:
            st.metric("Avg Reviews", f"{price_wars['no_of_ratings'].mean():,.0f}")
        
        with col_c:
            st.metric("Avg Rating", f"{price_wars['ratings'].mean():.2f}⭐")
        
        st.divider()
        
        # Category breakdown
        st.subheader("Price Wars by Category")
        category_wars = price_wars['main_category'].value_counts().head(15)
        
        fig = px.bar(
            x=category_wars.values,
            y=category_wars.index,
            orientation='h',
            title="Categories with Most Price Wars",
            labels={'x': 'Number of Products', 'y': 'Category'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Intensity scatter
        st.subheader("Competitive Intensity Map")
        
        sample = price_wars.head(100)  # Sample for performance
        fig = px.scatter(
            sample,
            x='discount_percent',
            y='no_of_ratings',
            size='competitive_intensity',
            color='ratings',
            hover_name='name',
            title="Price Wars: Discount vs Demand",
            labels={
                'discount_percent': 'Discount %',
                'no_of_ratings': 'Number of Reviews',
                'competitive_intensity': 'Intensity',
                'ratings': 'Rating'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data
        st.dataframe(price_wars, use_container_width=True, height=400)
    else:
        st.info("No price wars detected with current criteria")

with tab3:
    st.header("Premium Segment Identification")
    st.info("Find high-price, high-quality products with solid demand")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_pct = st.slider("Price Percentile", 50, 95, 75, key="premium_price")
    
    with col2:
        rating_min = st.slider("Min Rating", 3.0, 5.0, 4.2, 0.1, key="premium_rating")
    
    with col3:
        demand_pct = st.slider("Demand Percentile", 30, 90, 50, key="premium_demand")
    
    with st.spinner("Identifying premium products..."):
        premium = detector.identify_premium_segments(
            price_percentile=price_pct,
            rating_threshold=rating_min,
            demand_percentile=demand_pct
        )
    
    if not premium.empty:
        st.success(f"💎 Found {len(premium)} premium segment products")
        
        # Top premium products
        top_premium = premium.head(20)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_premium['name'],
            y=top_premium['discount_price'],
            name='Price',
            marker_color='gold'
        ))
        
        fig.update_layout(
            title="Top 20 Premium Products by Score",
            xaxis_title="Product",
            yaxis_title="Price (₹)",
            height=500
        )
        fig.update_xaxes(tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Premium score distribution
        st.subheader("Premium Score vs Price")
        
        fig = px.scatter(
            premium.head(100),
            x='discount_price',
            y='premium_score',
            size='no_of_ratings',
            color='ratings',
            hover_name='name',
            title="Premium Products: Price vs Quality Score",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data
        st.dataframe(premium, use_container_width=True, height=400)
        
        # Download
        csv = premium.to_csv(index=False)
        st.download_button("📥 Download Premium Products", csv, "premium_products.csv", "text/csv")
    else:
        st.info("No premium products found with current criteria")

with tab4:
    st.header("Underpriced Value Opportunities")
    st.info("Find high-quality products at below-market prices")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_pct = st.slider("Max Price Percentile", 10, 60, 40, key="value_price")
    
    with col2:
        rating_min = st.slider("Min Rating", 3.5, 5.0, 4.3, 0.1, key="value_rating")
    
    with col3:
        demand_pct = st.slider("Min Demand Percentile", 50, 95, 70, key="value_demand")
    
    with st.spinner("Finding underpriced gems..."):
        gems = detector.find_underpriced_gems(
            price_percentile=price_pct,
            rating_threshold=rating_min,
            demand_percentile=demand_pct
        )
    
    if not gems.empty:
        st.success(f"💰 Found {len(gems)} underpriced high-quality products!")
        
        # Best value products
        top_gems = gems.head(20)
        
        fig = px.scatter(
            top_gems,
            x='discount_price',
            y='ratings',
            size='no_of_ratings',
            color='value_score',
            hover_name='name',
            title="Top 20 Value Opportunities",
            labels={
                'discount_price': 'Price (₹)',
                'ratings': 'Rating',
                'no_of_ratings': 'Reviews',
                'value_score': 'Value Score'
            },
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        st.subheader("Value Opportunities by Category")
        cat_counts = gems['main_category'].value_counts().head(10)
        
        fig = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title="Distribution of Value Products"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data
        st.dataframe(gems, use_container_width=True, height=400)
        
        # Download
        csv = gems.to_csv(index=False)
        st.download_button("📥 Download Value Opportunities", csv, "value_opportunities.csv", "text/csv")
    else:
        st.info("No underpriced gems found with current criteria")

with tab5:
    st.header("Comprehensive Opportunity Matrix")
    
    with st.spinner("Creating opportunity matrix..."):
        df_matrix = detector.create_opportunity_matrix()
    
    if 'opportunity_type' in df_matrix.columns:
        # Distribution
        opp_counts = df_matrix['opportunity_type'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                x=opp_counts.values,
                y=opp_counts.index,
                orientation='h',
                title="Opportunity Type Distribution",
                labels={'x': 'Number of Products', 'y': 'Opportunity Type'},
                color=opp_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Summary Stats")
            for opp_type, count in opp_counts.items():
                st.metric(opp_type, f"{count:,}", f"{count/len(df_matrix)*100:.1f}%")
        
        # Category opportunities
        st.subheader("Opportunities by Category")
        
        category_opp = detector.analyze_category_opportunities()
        
        if not category_opp.empty:
            # Show only percentage columns
            pct_cols = [col for col in category_opp.columns if '_pct' in col and col != 'total_products']
            display_cols = ['main_category', 'total_products'] + pct_cols
            
            st.dataframe(
                category_opp[display_cols].head(20),
                use_container_width=True,
                height=400
            )
        
        # Interactive exploration
        st.subheader("Explore by Opportunity Type")
        
        selected_type = st.selectbox(
            "Select Opportunity Type",
            options=sorted(df_matrix['opportunity_type'].unique())
        )
        
        filtered = df_matrix[df_matrix['opportunity_type'] == selected_type]
        
        st.info(f"Showing {len(filtered):,} products in '{selected_type}' category")
        
        display_cols = ['name', 'main_category', 'discount_price', 'ratings', 'no_of_ratings', 'discount_percent']
        available_cols = [col for col in display_cols if col in filtered.columns]
        
        st.dataframe(
            filtered[available_cols].head(100),
            use_container_width=True,
            height=400
        )
        
        # Download
        csv = filtered[available_cols].to_csv(index=False)
        st.download_button(
            f"📥 Download {selected_type} Products",
            csv,
            f"{selected_type.lower().replace(' ', '_')}_products.csv",
            "text/csv"
        )
