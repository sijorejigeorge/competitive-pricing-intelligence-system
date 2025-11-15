"""
Price Elasticity & Optimization Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.elasticity import ElasticityModel

st.set_page_config(page_title="Price Elasticity", page_icon="🔮", layout="wide")

st.title("🔮 Price Elasticity & Revenue Optimization")

if 'df' not in st.session_state:
    st.warning("⚠️ Please load data from the main page first")
    st.stop()

df = st.session_state.df

# Initialize model
model = ElasticityModel(df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overall Elasticity",
    "📊 Category Analysis",
    "💰 Price Simulator",
    "🎯 Optimization"
])

with tab1:
    st.header("Overall Price Elasticity")
    
    st.info("""
    **Price Elasticity of Demand** measures how demand changes when price changes.
    
    - **Elasticity < -1:** Elastic (demand is price-sensitive)
    - **Elasticity > -1:** Inelastic (demand is price-insensitive)
    - **Elasticity = -1:** Unit elastic
    """)
    
    with st.spinner("Calculating overall elasticity..."):
        overall_results = model.estimate_price_elasticity()
    
    if 'error' not in overall_results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Elasticity", f"{overall_results['elasticity']:.3f}")
        
        with col2:
            st.metric("R² Score", f"{overall_results['r_squared']:.3f}")
        
        with col3:
            st.metric("Sample Size", f"{overall_results['sample_size']:,}")
        
        with col4:
            interpretation = overall_results['interpretation']
            st.metric("Type", interpretation.split('(')[0].strip())
        
        st.divider()
        
        # Interpretation
        elasticity = overall_results['elasticity']
        
        if elasticity < -1:
            st.success(f"""
            **🎯 Elastic Demand** (elasticity = {elasticity:.3f})
            
            Demand is highly sensitive to price changes. Recommendations:
            - Consider **price reductions** to increase volume
            - **Promotions and discounts** are effective
            - Focus on **market share** over margin
            - **Competitive pricing** is critical
            """)
        elif elasticity > -1 and elasticity < 0:
            st.info(f"""
            **💎 Inelastic Demand** (elasticity = {elasticity:.3f})
            
            Demand is relatively insensitive to price. Recommendations:
            - Opportunity for **price increases**
            - Focus on **margin optimization**
            - **Premium positioning** possible
            - **Value-added features** can justify higher prices
            """)
        else:
            st.warning(f"""
            **⚠️ Unusual Elasticity** (elasticity = {elasticity:.3f})
            
            Positive elasticity suggests higher prices lead to higher demand (prestige goods).
            This may indicate data quality issues or luxury segment behavior.
            """)
    else:
        st.error("Insufficient data to calculate elasticity")

with tab2:
    st.header("Elasticity by Category")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_n = st.slider("Number of categories", 5, 30, 15, key="cat_elastic")
        min_products = st.slider("Min products", 20, 200, 50, 10, key="min_prod")
    
    with st.spinner("Analyzing categories..."):
        category_results = model.estimate_by_category(top_n=top_n, min_products=min_products)
    
    if not category_results.empty:
        st.success(f"Analyzed {len(category_results)} categories")
        
        # Visualization
        fig = px.bar(
            category_results,
            x='category',
            y='elasticity',
            color='interpretation',
            title="Price Elasticity by Category",
            labels={'elasticity': 'Elasticity Coefficient', 'category': 'Category'},
            color_discrete_map={
                'Elastic (demand is price-sensitive)': '#ff6b6b',
                'Inelastic (demand is price-insensitive)': '#4ecdc4',
                'Unit elastic': '#ffe66d'
            }
        )
        fig.add_hline(y=-1, line_dash="dash", line_color="gray", 
                     annotation_text="Unit Elastic")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # R-squared distribution
        st.subheader("Model Quality (R² Scores)")
        
        fig = px.scatter(
            category_results,
            x='category',
            y='r_squared',
            size='sample_size',
            color='r_squared',
            title="Model Fit Quality by Category",
            labels={'r_squared': 'R² Score', 'category': 'Category'},
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0.3, line_dash="dash", line_color="gray", 
                     annotation_text="Acceptable Fit")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Detailed Results")
        st.dataframe(category_results, use_container_width=True)
        
        # Download
        csv = category_results.to_csv(index=False)
        st.download_button(
            "📥 Download Elasticity Analysis",
            csv,
            "category_elasticity.csv",
            "text/csv"
        )
    else:
        st.info("No categories meet the minimum product threshold")

with tab3:
    st.header("Price Change Simulator")
    
    st.info("Simulate the impact of price changes on demand and revenue")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        base_price = st.number_input(
            "Current Price (₹)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100
        )
        
        elasticity_input = st.number_input(
            "Price Elasticity",
            min_value=-10.0,
            max_value=2.0,
            value=-1.5,
            step=0.1,
            help="Use estimates from elasticity analysis"
        )
        
        price_changes = st.text_input(
            "Price Changes (%) - comma separated",
            value="-30,-20,-10,-5,0,5,10,20,30,50"
        )
        
        # Parse price changes
        try:
            price_change_list = [float(x.strip()) for x in price_changes.split(',')]
        except:
            st.error("Invalid price changes format")
            price_change_list = [-30, -20, -10, 0, 10, 20, 30]
    
    with col2:
        with st.spinner("Running simulation..."):
            simulation = model.simulate_price_changes(
                base_price=base_price,
                elasticity=elasticity_input,
                price_changes=price_change_list
            )
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=simulation['price_change_pct'],
            y=simulation['revenue_change_pct'],
            mode='lines+markers',
            name='Revenue Change',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Revenue Impact of Price Changes",
            xaxis_title="Price Change (%)",
            yaxis_title="Revenue Change (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.subheader("Simulation Results")
    
    # Highlight optimal
    optimal_idx = simulation['new_revenue_index'].idxmax()
    simulation['is_optimal'] = simulation.index == optimal_idx
    
    # Style the dataframe
    def highlight_optimal(row):
        if row['is_optimal']:
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        simulation.style.apply(highlight_optimal, axis=1),
        use_container_width=True
    )
    
    # Insights
    optimal_row = simulation.loc[optimal_idx]
    st.success(f"""
    **🎯 Optimal Price Point**
    
    - **Price Change:** {optimal_row['price_change_pct']:.0f}%
    - **New Price:** ₹{optimal_row['new_price']:.0f}
    - **Expected Revenue Change:** {optimal_row['revenue_change_pct']:.1f}%
    - **Expected Demand Change:** {optimal_row['demand_change_pct']:.1f}%
    """)

with tab4:
    st.header("Revenue Optimization Tool")
    
    st.info("Find the price that maximizes revenue based on elasticity")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Product Parameters")
        
        current_price = st.number_input(
            "Current Price (₹)",
            min_value=100,
            max_value=100000,
            value=2000,
            step=100,
            key="opt_price"
        )
        
        elasticity_val = st.number_input(
            "Price Elasticity",
            min_value=-10.0,
            max_value=2.0,
            value=-1.8,
            step=0.1,
            key="opt_elasticity"
        )
        
        min_change = st.slider("Min Price Change (%)", -80, 0, -50)
        max_change = st.slider("Max Price Change (%)", 0, 200, 100)
        
        if st.button("🔍 Find Optimal Price", type="primary"):
            with st.spinner("Optimizing..."):
                optimal_result = model.find_revenue_maximizing_price(
                    base_price=current_price,
                    elasticity=elasticity_val,
                    min_price_change=min_change,
                    max_price_change=max_change
                )
                
                st.session_state.optimal_result = optimal_result
    
    with col2:
        if 'optimal_result' in st.session_state:
            result = st.session_state.optimal_result
            
            st.subheader("💰 Optimization Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Optimal Price",
                    f"₹{result['optimal_price']:,.0f}",
                    delta=f"{result['price_change_from_current']:.1f}%"
                )
            
            with col_b:
                st.metric(
                    "Revenue Change",
                    f"{result['expected_revenue_change']:.1f}%",
                    delta="Expected"
                )
            
            with col_c:
                st.metric(
                    "Demand Change",
                    f"{result['expected_demand_change']:.1f}%",
                    delta="Expected"
                )
            
            st.divider()
            
            # Recommendations
            if result['price_change_from_current'] > 0:
                st.success(f"""
                **📈 Recommendation: Increase Price**
                
                - Current price of ₹{current_price:,.0f} is below optimal
                - Increase to ₹{result['optimal_price']:,.0f} (+{result['price_change_from_current']:.1f}%)
                - Expected revenue gain: {result['expected_revenue_change']:.1f}%
                - With elasticity of {elasticity_val:.2f}, demand is relatively inelastic
                - Higher margins can offset volume decline
                """)
            elif result['price_change_from_current'] < 0:
                st.info(f"""
                **📉 Recommendation: Decrease Price**
                
                - Current price of ₹{current_price:,.0f} is above optimal
                - Decrease to ₹{result['optimal_price']:,.0f} ({result['price_change_from_current']:.1f}%)
                - Expected revenue gain: {result['expected_revenue_change']:.1f}%
                - With elasticity of {elasticity_val:.2f}, demand is price-sensitive
                - Volume increase will offset margin reduction
                """)
            else:
                st.success("Current price is already optimal! 🎯")
        else:
            st.info("👈 Enter parameters and click 'Find Optimal Price' to see recommendations")
    
    st.divider()
    
    # Pricing strategies
    st.header("📚 Pricing Strategy Guidelines")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("""
        ### Elastic Demand (|E| > 1)
        
        **Characteristics:**
        - Customers are price-sensitive
        - Many substitutes available
        - Non-essential products
        - Highly competitive markets
        
        **Strategies:**
        - ✅ Competitive pricing
        - ✅ Volume-based discounts
        - ✅ Promotional pricing
        - ✅ Bundle offers
        - ❌ Avoid premium pricing
        """)
    
    with col_s2:
        st.markdown("""
        ### Inelastic Demand (|E| < 1)
        
        **Characteristics:**
        - Customers less price-sensitive
        - Few substitutes
        - Essential or unique products
        - Strong brand loyalty
        
        **Strategies:**
        - ✅ Premium pricing
        - ✅ Value-based pricing
        - ✅ Feature differentiation
        - ✅ Brand positioning
        - ❌ Avoid price wars
        """)

st.divider()
st.caption("💡 Tip: Combine elasticity insights with competitive analysis for best results")
