"""
Main Streamlit Dashboard for Competitive Intelligence System
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_and_clean_data, get_data_summary
from src.utils.preprocessor import DataPreprocessor
from src.utils.config import DASHBOARD_TITLE, PAGE_ICON, LAYOUT

# Page config
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.2rem;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    .insight-box strong {
        color: #1e40af;
        font-size: 1.05rem;
    }
    
    /* Section headers */
    .stMarkdown h3 {
        color: #1e3a8a !important;
        background-color: transparent !important;
        padding: 0.5rem 0 !important;
        margin-bottom: 1rem !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a8a !important;
    }
    
    /* Override Streamlit's default dark background on headers */
    [data-testid="stMarkdownContainer"] h3 {
        background: none !important;
    }
    
    /* Force light text on main area */
    .main .stMarkdown h1,
    .main .stMarkdown h2,
    .main .stMarkdown h3,
    .main .stMarkdown h4 {
        color: #1e3a8a !important;
        background: transparent !important;
    }
    
    /* Section dividers */
    hr {
        border-color: #e5e7eb !important;
        margin: 2rem 0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Charts */
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(sample_frac=0.1):
    """Load and cache data"""
    try:
        df = load_and_clean_data(sample_frac=sample_frac)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def preprocess_data(df):
    """Preprocess and add features"""
    try:
        preprocessor = DataPreprocessor(df)
        df_processed = preprocessor.get_processed_data()
        return df_processed
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return df


def main():
    """Main dashboard"""
    
    # Header
    st.markdown(f'<div class="main-header">{DASHBOARD_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">📊 Real-time competitive intelligence across 300K+ products</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # Data loading options
        st.subheader("📦 Data Configuration")
        sample_size = st.slider(
            "Sample Size (%)", 
            min_value=1, 
            max_value=100, 
            value=10,
            help="💡 Start with 10% for quick loading. Increase for more accurate insights."
        )
        
        if st.button("🔄 Reload Data", type="primary", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        if 'df' in st.session_state:
            df_stats = st.session_state.df
            st.subheader("📊 Current Dataset")
            st.metric("Products", f"{len(df_stats):,}")
            st.metric("Categories", f"{df_stats['main_category'].nunique()}")
            st.metric("Avg Price", f"₹{df_stats['discount_price'].mean():,.0f}")
        
        st.markdown("---")
        
        # Navigation help
        st.subheader("🧭 Navigation")
        st.markdown("""
        Use the sidebar pages to explore:
        
        **💰 Pricing Intelligence**  
        Price benchmarks & strategies
        
        **📊 Demand Analysis**  
        Market trends & popularity
        
        **🎯 Market Opportunities**  
        Gaps & competitive insights
        
        **🔮 Price Elasticity**  
        Revenue optimization
        """)
        
        st.markdown("---")
        st.caption("🚀 Built with Streamlit & Python")
    
    # Load data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner(f"Loading {sample_size}% of data..."):
            df = load_data(sample_frac=sample_size/100)
            if df is not None:
                df_processed = preprocess_data(df)
                st.session_state.df = df_processed
                st.session_state.data_loaded = True
                st.success(f"✓ Loaded {len(df_processed):,} products")
            else:
                st.error("Failed to load data. Please run download_dataset.py first.")
                return
    
    df = st.session_state.df
    
    # Main content
    st.markdown('<h3 style="color: #1e3a8a; background: transparent;">📈 Executive Overview</h3>', unsafe_allow_html=True)
    st.markdown("Real-time metrics from your competitive intelligence analysis")
    st.markdown("")
    
    # Key metrics with better spacing
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🛍️ Total Products",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="📂 Categories",
            value=f"{df['main_category'].nunique():,}",
            delta=None
        )
    
    with col3:
        avg_price = df['discount_price'].mean()
        st.metric(
            label="💵 Avg Price",
            value=f"₹{avg_price:,.0f}",
            delta=None
        )
    
    with col4:
        avg_discount = df['discount_percent'].mean()
        st.metric(
            label="🏷️ Avg Discount",
            value=f"{avg_discount:.1f}%",
            delta=None
        )
    
    with col5:
        avg_rating = df['ratings'].mean()
        st.metric(
            label="⭐ Avg Rating",
            value=f"{avg_rating:.2f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Two-column layout
    col_left, col_right = st.columns([2.5, 1.5])
    
    with col_left:
        st.markdown('<h3 style="color: #1e3a8a; background: transparent;">📊 Top Categories by Product Count</h3>', unsafe_allow_html=True)
        
        # Top categories by product count
        top_categories = df['main_category'].value_counts().head(15)
        
        import plotly.express as px
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            labels={'x': 'Number of Products', 'y': 'Category'},
            color=top_categories.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            showlegend=False,
            height=500,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"📌 Showing top {len(top_categories)} categories out of {df['main_category'].nunique()} total")
    
    with col_right:
        st.markdown('<h3 style="color: #1e3a8a; background: transparent;">💡 Key Insights</h3>', unsafe_allow_html=True)
        
        # Calculate insights
        total_reviews = df['no_of_ratings'].sum()
        products_with_discount = (df['discount_percent'] > 0).sum()
        high_rated = (df['ratings'] >= 4.0).sum()
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📦 Coverage</strong><br>
        <span style="color: #1e3a8a;">{len(df):,} products across {df['main_category'].nunique()} categories</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>💰 Discounts</strong><br>
        <span style="color: #1e3a8a;">{products_with_discount:,} products ({products_with_discount/len(df)*100:.1f}%) on sale</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>⭐ High Quality</strong><br>
        <span style="color: #1e3a8a;">{high_rated:,} products ({high_rated/len(df)*100:.1f}%) rated 4.0+</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📈 Engagement</strong><br>
        <span style="color: #1e3a8a;">{total_reviews:,.0f} total reviews</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data explorer
    st.markdown('<h3 style="color: #1e3a8a; background: transparent;">🔍 Product Explorer</h3>', unsafe_allow_html=True)
    st.markdown("Filter and explore products based on your criteria")
    st.markdown("")
    
    # Filters in expandable section
    with st.expander("🎯 Filter Options", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            selected_categories = st.multiselect(
                "📂 Categories",
                options=sorted(df['main_category'].unique()),
                default=None,
                help="Select one or more categories to filter"
            )
        
        with col_f2:
            price_range = st.slider(
                "💰 Price Range (₹)",
                min_value=int(df['discount_price'].min()),
                max_value=int(df['discount_price'].max()),
                value=(int(df['discount_price'].min()), int(df['discount_price'].max())),
                help="Set minimum and maximum price"
            )
        
        with col_f3:
            min_rating = st.slider(
                "⭐ Minimum Rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                help="Filter products with ratings above this value"
            )
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_categories:
        df_filtered = df_filtered[df_filtered['main_category'].isin(selected_categories)]
    
    df_filtered = df_filtered[
        (df_filtered['discount_price'] >= price_range[0]) &
        (df_filtered['discount_price'] <= price_range[1]) &
        (df_filtered['ratings'] >= min_rating)
    ]
    
    # Results summary
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("📊 Filtered Products", f"{len(df_filtered):,}")
    with col_summary2:
        st.metric("📈 Avg Price (Filtered)", f"₹{df_filtered['discount_price'].mean():,.0f}")
    with col_summary3:
        st.metric("⭐ Avg Rating (Filtered)", f"{df_filtered['ratings'].mean():.2f}")
    
    st.markdown("")
    
    # Display data
    display_cols = ['name', 'main_category', 'discount_price', 'actual_price', 
                   'discount_percent', 'ratings', 'no_of_ratings']
    
    available_cols = [col for col in display_cols if col in df_filtered.columns]
    
    st.dataframe(
        df_filtered[available_cols].head(100),
        use_container_width=True,
        height=450,
        hide_index=True
    )
    
    # Download button
    col_download1, col_download2 = st.columns([1, 4])
    with col_download1:
        csv = df_filtered[available_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="filtered_products.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Next steps
    st.markdown('<h3 style="color: #1e3a8a; background: transparent;">🚀 Explore Advanced Analytics</h3>', unsafe_allow_html=True)
    st.markdown("Navigate to specialized analysis pages using the sidebar")
    st.markdown("")
    
    col_n1, col_n2, col_n3 = st.columns(3)
    
    with col_n1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid #f59e0b; height: 180px;">
        <h4 style="color: #92400e; margin-top: 0;">💰 Pricing Intelligence</h4>
        <p style="color: #78350f; margin-bottom: 0.5rem;">
        • Price benchmarks by category<br>
        • Brand premium analysis<br>
        • Competitive positioning<br>
        • Discount strategies
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_n2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid #3b82f6; height: 180px;">
        <h4 style="color: #1e40af; margin-top: 0;">📊 Demand Analysis</h4>
        <p style="color: #1e3a8a; margin-bottom: 0.5rem;">
        • Demand segmentation<br>
        • Popularity trends<br>
        • High-demand products<br>
        • Market opportunities
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_n3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid #10b981; height: 180px;">
        <h4 style="color: #065f46; margin-top: 0;">🎯 Market Opportunities</h4>
        <p style="color: #064e3b; margin-bottom: 0.5rem;">
        • Market gap detection<br>
        • Price war identification<br>
        • Premium segments<br>
        • Value opportunities
        </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
