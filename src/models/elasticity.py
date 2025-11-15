"""
Price elasticity modeling and optimization
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ElasticityModel:
    """
    Estimate price elasticity of demand and optimize pricing
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize elasticity model
        
        Args:
            df: Preprocessed dataframe with pricing and demand data
        """
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist"""
        required_cols = ['discount_price', 'no_of_ratings']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            print(f"Warning: Missing columns {missing}")
    
    def estimate_price_elasticity(self, category: Optional[str] = None, 
                                  use_log: bool = True) -> Dict:
        """
        Estimate price elasticity of demand
        
        Elasticity = % change in demand / % change in price
        
        Args:
            category: Specific category to analyze (None for overall)
            use_log: Use log-log model (better for elasticity)
        
        Returns:
            dict: Elasticity results
        """
        # Filter data
        if category:
            df = self.df[self.df['main_category'] == category].copy()
        else:
            df = self.df.copy()
        
        # Clean data
        df = df[['discount_price', 'no_of_ratings']].dropna()
        df = df[(df['discount_price'] > 0) & (df['no_of_ratings'] > 0)]
        
        if len(df) < 10:
            return {'error': 'Insufficient data'}
        
        # Prepare data
        X = df['discount_price'].values.reshape(-1, 1)
        y = df['no_of_ratings'].values
        
        if use_log:
            # Log-log model: log(Q) = a + b*log(P)
            # Elasticity = b (constant elasticity)
            X = np.log(X)
            y = np.log(y)
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        elasticity = model.coef_[0]
        r_squared = model.score(X, y)
        
        # Interpret elasticity
        if abs(elasticity) > 1:
            interpretation = 'Elastic (demand is price-sensitive)'
        elif abs(elasticity) < 1:
            interpretation = 'Inelastic (demand is price-insensitive)'
        else:
            interpretation = 'Unit elastic'
        
        results = {
            'elasticity': round(elasticity, 3),
            'r_squared': round(r_squared, 3),
            'interpretation': interpretation,
            'sample_size': len(df),
            'model_type': 'log-log' if use_log else 'linear',
            'category': category if category else 'Overall'
        }
        
        return results
    
    def estimate_by_category(self, top_n: int = 20, min_products: int = 50) -> pd.DataFrame:
        """
        Estimate elasticity for multiple categories
        
        Args:
            top_n: Number of top categories to analyze
            min_products: Minimum products required per category
        
        Returns:
            pd.DataFrame: Elasticity by category
        """
        # Get top categories
        category_counts = self.df['main_category'].value_counts()
        top_categories = category_counts[category_counts >= min_products].head(top_n).index
        
        results = []
        
        for category in top_categories:
            result = self.estimate_price_elasticity(category=category)
            if 'error' not in result:
                results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('elasticity', ascending=True)
        
        return results_df
    
    def calculate_optimal_price(self, current_price: float, 
                               elasticity: float,
                               cost: Optional[float] = None,
                               margin_target: Optional[float] = None) -> Dict:
        """
        Calculate optimal price to maximize revenue or profit
        
        Args:
            current_price: Current product price
            elasticity: Price elasticity (from estimate_price_elasticity)
            cost: Product cost (for profit optimization)
            margin_target: Target profit margin
        
        Returns:
            dict: Optimal pricing recommendations
        """
        # Revenue maximization (when cost unknown)
        # Optimal markup = 1 / (1 + elasticity)
        if elasticity >= -1:
            # Inelastic or positive elasticity - can increase price
            optimal_markup = 1.5  # Conservative increase
            recommendation = "Consider gradual price increases"
        else:
            # Elastic demand
            optimal_markup = 1 / (1 - 1/elasticity)
            recommendation = "Optimize for volume"
        
        optimal_price_revenue = current_price * optimal_markup
        
        results = {
            'current_price': current_price,
            'elasticity': elasticity,
            'optimal_price_revenue_max': round(optimal_price_revenue, 2),
            'price_change_pct': round((optimal_price_revenue - current_price) / current_price * 100, 2),
            'recommendation': recommendation
        }
        
        # Profit optimization if cost known
        if cost is not None:
            # Optimal price = cost / (1 + 1/elasticity)
            if elasticity < -1:
                optimal_price_profit = cost / (1 + 1/elasticity)
                results['optimal_price_profit_max'] = round(optimal_price_profit, 2)
                results['profit_margin'] = round((optimal_price_profit - cost) / optimal_price_profit * 100, 2)
        
        return results
    
    def simulate_price_changes(self, base_price: float, 
                              elasticity: float,
                              price_changes: List[float] = None) -> pd.DataFrame:
        """
        Simulate impact of price changes on demand and revenue
        
        Args:
            base_price: Current price
            elasticity: Price elasticity
            price_changes: List of price change percentages to simulate
        
        Returns:
            pd.DataFrame: Simulation results
        """
        if price_changes is None:
            price_changes = [-30, -20, -10, -5, 0, 5, 10, 20, 30, 50]
        
        results = []
        base_demand = 100  # Normalize to 100
        
        for change_pct in price_changes:
            new_price = base_price * (1 + change_pct / 100)
            
            # Calculate demand change using elasticity
            # % change in Q = elasticity × % change in P
            demand_change_pct = elasticity * change_pct
            new_demand = base_demand * (1 + demand_change_pct / 100)
            
            # Calculate revenue
            base_revenue = base_price * base_demand
            new_revenue = new_price * new_demand
            revenue_change_pct = (new_revenue - base_revenue) / base_revenue * 100
            
            results.append({
                'price_change_pct': change_pct,
                'new_price': round(new_price, 2),
                'demand_change_pct': round(demand_change_pct, 2),
                'new_demand_index': round(new_demand, 2),
                'revenue_change_pct': round(revenue_change_pct, 2),
                'new_revenue_index': round(new_revenue, 2)
            })
        
        return pd.DataFrame(results)
    
    def find_revenue_maximizing_price(self, base_price: float, 
                                     elasticity: float,
                                     min_price_change: float = -50,
                                     max_price_change: float = 100) -> Dict:
        """
        Find the price that maximizes revenue
        
        Args:
            base_price: Current price
            elasticity: Price elasticity
            min_price_change: Minimum price change % to consider
            max_price_change: Maximum price change % to consider
        
        Returns:
            dict: Optimal pricing for revenue maximization
        """
        # Simulate many price points
        price_changes = np.linspace(min_price_change, max_price_change, 100)
        simulation = self.simulate_price_changes(base_price, elasticity, price_changes.tolist())
        
        # Find maximum revenue
        max_revenue_idx = simulation['new_revenue_index'].idxmax()
        optimal_result = simulation.loc[max_revenue_idx]
        
        return {
            'optimal_price': optimal_result['new_price'],
            'price_change_from_current': optimal_result['price_change_pct'],
            'expected_demand_change': optimal_result['demand_change_pct'],
            'expected_revenue_change': optimal_result['revenue_change_pct'],
            'elasticity': elasticity
        }
    
    def analyze_price_sensitivity_by_segment(self, segment_col: str = 'price_tier') -> pd.DataFrame:
        """
        Analyze price sensitivity across different segments
        
        Args:
            segment_col: Column to segment by
        
        Returns:
            pd.DataFrame: Elasticity by segment
        """
        if segment_col not in self.df.columns:
            print(f"Column {segment_col} not found")
            return pd.DataFrame()
        
        results = []
        
        for segment in self.df[segment_col].dropna().unique():
            df_segment = self.df[self.df[segment_col] == segment]
            
            # Clean data
            df_clean = df_segment[['discount_price', 'no_of_ratings']].dropna()
            df_clean = df_clean[(df_clean['discount_price'] > 0) & (df_clean['no_of_ratings'] > 0)]
            
            if len(df_clean) >= 20:
                # Calculate correlation (simple elasticity proxy)
                corr, pval = stats.spearmanr(
                    df_clean['discount_price'],
                    df_clean['no_of_ratings']
                )
                
                results.append({
                    'segment': segment,
                    'correlation': round(corr, 3),
                    'p_value': round(pval, 4),
                    'sample_size': len(df_clean),
                    'avg_price': round(df_clean['discount_price'].mean(), 2),
                    'price_sensitivity': 'High' if abs(corr) > 0.3 else 'Low'
                })
        
        results_df = pd.DataFrame(results)
        return results_df.sort_values('correlation', ascending=True)
    
    def visualize_price_demand_relationship(self, category: Optional[str] = None, 
                                           sample_size: int = 1000,
                                           figsize: tuple = (14, 6)):
        """
        Visualize price-demand relationship
        
        Args:
            category: Specific category to analyze
            sample_size: Number of points to plot
            figsize: Figure size
        
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Filter data
        if category:
            df = self.df[self.df['main_category'] == category].copy()
        else:
            df = self.df.copy()
        
        df = df[['discount_price', 'no_of_ratings', 'ratings']].dropna()
        df = df[(df['discount_price'] > 0) & (df['no_of_ratings'] > 0)]
        
        # Sample if too large
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Scatter plot with trend line
        axes[0].scatter(df['discount_price'], df['no_of_ratings'], 
                       alpha=0.5, s=30, c='steelblue')
        
        # Add trend line
        z = np.polyfit(np.log(df['discount_price']), np.log(df['no_of_ratings']), 1)
        p = np.poly1d(z)
        
        x_trend = np.linspace(df['discount_price'].min(), df['discount_price'].max(), 100)
        y_trend = np.exp(p(np.log(x_trend)))
        
        axes[0].plot(x_trend, y_trend, "r--", linewidth=2, label=f'Trend (elasticity ≈ {z[0]:.2f})')
        axes[0].set_xlabel('Price (₹)')
        axes[0].set_ylabel('Number of Reviews (Demand Proxy)')
        axes[0].set_title('Price vs Demand Relationship')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Price bins vs average demand
        df['price_bin'] = pd.qcut(df['discount_price'], q=10, duplicates='drop')
        bin_stats = df.groupby('price_bin').agg({
            'no_of_ratings': 'mean',
            'ratings': 'mean'
        }).reset_index()
        
        bin_stats['price_midpoint'] = bin_stats['price_bin'].apply(lambda x: x.mid)
        
        ax2 = axes[1]
        ax2.bar(range(len(bin_stats)), bin_stats['no_of_ratings'], 
               color='coral', alpha=0.7, label='Avg Reviews')
        ax2.set_xlabel('Price Decile (1=Lowest, 10=Highest)')
        ax2.set_ylabel('Average Reviews', color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax2.set_title('Demand by Price Segment')
        
        # Add rating on secondary axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(bin_stats)), bin_stats['ratings'], 
                     'bo-', linewidth=2, label='Avg Rating')
        ax2_twin.set_ylabel('Average Rating', color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        ax2_twin.set_ylim(0, 5)
        
        plt.tight_layout()
        return fig
    
    def get_elasticity_summary(self, top_n_categories: int = 10) -> Dict:
        """
        Get comprehensive elasticity summary
        
        Args:
            top_n_categories: Number of top categories to analyze
        
        Returns:
            dict: Elasticity summary
        """
        # Overall elasticity
        overall = self.estimate_price_elasticity()
        
        # By category
        by_category = self.estimate_by_category(top_n=top_n_categories)
        
        summary = {
            'overall_elasticity': overall.get('elasticity'),
            'overall_interpretation': overall.get('interpretation'),
            'most_elastic_category': None,
            'most_inelastic_category': None,
            'avg_elasticity': None
        }
        
        if not by_category.empty:
            summary['most_elastic_category'] = by_category.iloc[0]['category']
            summary['most_inelastic_category'] = by_category.iloc[-1]['category']
            summary['avg_elasticity'] = round(by_category['elasticity'].mean(), 3)
        
        return summary
