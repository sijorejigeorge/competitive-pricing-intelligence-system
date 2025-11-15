"""
Demand modeling and popularity analysis module
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DemandModeler:
    """
    Model and analyze product demand using ratings as a proxy
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize demand modeler
        
        Args:
            df: Preprocessed dataframe with ratings data
        """
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist"""
        if 'no_of_ratings' not in self.df.columns:
            print("Warning: 'no_of_ratings' column not found")
    
    def calculate_popularity_index(self, base: float = 10.0) -> pd.Series:
        """
        Calculate popularity index using log transformation of review count
        
        Args:
            base: Logarithm base
        
        Returns:
            pd.Series: Popularity index scores
        """
        if 'no_of_ratings' not in self.df.columns:
            return pd.Series(np.nan, index=self.df.index)
        
        popularity = np.log(self.df['no_of_ratings'].fillna(0) + 1) / np.log(base)
        return popularity
    
    def calculate_quality_score(self) -> pd.Series:
        """
        Calculate quality score (rating × popularity)
        
        Returns:
            pd.Series: Quality scores
        """
        popularity = self.calculate_popularity_index()
        
        if 'ratings' in self.df.columns:
            quality = self.df['ratings'].fillna(0) * popularity
            return quality
        else:
            return popularity
    
    def segment_by_demand(self, rating_col: str = 'no_of_ratings') -> pd.DataFrame:
        """
        Segment products by demand level
        
        Args:
            rating_col: Column with rating count
        
        Returns:
            pd.DataFrame: Segmented data
        """
        df = self.df.copy()
        
        if rating_col not in df.columns:
            print(f"Column {rating_col} not found")
            return df
        
        # Define demand segments
        df['demand_segment'] = pd.cut(
            df[rating_col],
            bins=[-1, 0, 10, 100, 1000, 10000, float('inf')],
            labels=['No Demand', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate segment statistics
        segment_stats = df.groupby('demand_segment').agg({
            'name': 'count',
            'discount_price': 'mean',
            'ratings': 'mean',
            rating_col: ['mean', 'sum']
        }).round(2)
        
        print("\n📊 Demand Segmentation Summary:")
        print(segment_stats)
        
        return df
    
    def identify_high_demand_products(self, percentile: float = 90.0, 
                                     min_rating: float = 4.0) -> pd.DataFrame:
        """
        Identify high-demand products (high reviews + good ratings)
        
        Args:
            percentile: Percentile threshold for high demand
            min_rating: Minimum rating threshold
        
        Returns:
            pd.DataFrame: High-demand products
        """
        if 'no_of_ratings' not in self.df.columns:
            print("'no_of_ratings' column not found")
            return pd.DataFrame()
        
        threshold = self.df['no_of_ratings'].quantile(percentile / 100)
        
        high_demand = self.df[
            (self.df['no_of_ratings'] >= threshold) & 
            (self.df['ratings'] >= min_rating)
        ].copy()
        
        # Sort by demand
        high_demand = high_demand.sort_values('no_of_ratings', ascending=False)
        
        print(f"\n🔥 Found {len(high_demand)} high-demand products")
        print(f"   (>{percentile}th percentile, rating >={min_rating})")
        
        return high_demand[['name', 'main_category', 'ratings', 'no_of_ratings', 
                           'discount_price', 'discount_percent']]
    
    def analyze_demand_by_category(self, category_col: str = 'main_category') -> pd.DataFrame:
        """
        Analyze demand patterns by category
        
        Args:
            category_col: Category column
        
        Returns:
            pd.DataFrame: Category-level demand insights
        """
        category_demand = self.df.groupby(category_col).agg({
            'no_of_ratings': ['sum', 'mean', 'median', 'max'],
            'ratings': 'mean',
            'discount_price': 'mean',
            'name': 'count'
        }).round(2)
        
        # Flatten columns
        category_demand.columns = ['_'.join(col).strip() for col in category_demand.columns]
        category_demand = category_demand.reset_index()
        
        # Rename
        category_demand.columns = [
            'category', 
            'total_reviews', 'avg_reviews', 'median_reviews', 'max_reviews',
            'avg_rating', 'avg_price', 'product_count'
        ]
        
        # Calculate demand intensity (total reviews / product count)
        category_demand['demand_intensity'] = (
            category_demand['total_reviews'] / category_demand['product_count']
        ).round(2)
        
        # Calculate market share proxy (% of total reviews)
        total_market_reviews = category_demand['total_reviews'].sum()
        category_demand['market_share_proxy'] = (
            (category_demand['total_reviews'] / total_market_reviews) * 100
        ).round(2)
        
        # Sort by total reviews
        category_demand = category_demand.sort_values('total_reviews', ascending=False)
        
        return category_demand
    
    def find_low_competition_opportunities(self, demand_threshold: float = 0.3, 
                                          quality_threshold: float = 4.0) -> pd.DataFrame:
        """
        Find categories with low competition but decent demand
        
        Args:
            demand_threshold: Minimum demand intensity
            quality_threshold: Minimum quality score
        
        Returns:
            pd.DataFrame: Opportunity categories
        """
        category_demand = self.analyze_demand_by_category()
        
        # Low competition = high demand intensity but fewer products
        median_products = category_demand['product_count'].median()
        median_demand = category_demand['demand_intensity'].median()
        
        opportunities = category_demand[
            (category_demand['product_count'] < median_products) & 
            (category_demand['demand_intensity'] > median_demand * demand_threshold) &
            (category_demand['avg_rating'] >= quality_threshold)
        ]
        
        print(f"\n💡 Found {len(opportunities)} low-competition opportunity categories")
        
        return opportunities
    
    def calculate_demand_price_correlation(self, by_category: bool = True) -> Dict:
        """
        Calculate correlation between demand and price
        
        Args:
            by_category: Whether to calculate by category
        
        Returns:
            dict: Correlation results
        """
        results = {}
        
        # Overall correlation
        if 'no_of_ratings' in self.df.columns and 'discount_price' in self.df.columns:
            df_clean = self.df[['no_of_ratings', 'discount_price']].dropna()
            
            if len(df_clean) > 0:
                corr, pvalue = stats.spearmanr(
                    df_clean['no_of_ratings'], 
                    df_clean['discount_price']
                )
                
                results['overall'] = {
                    'correlation': round(corr, 3),
                    'p_value': round(pvalue, 4),
                    'interpretation': 'positive' if corr > 0 else 'negative',
                    'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
                }
        
        # By category
        if by_category and 'main_category' in self.df.columns:
            category_corrs = []
            
            for category in self.df['main_category'].unique():
                df_cat = self.df[self.df['main_category'] == category]
                df_cat_clean = df_cat[['no_of_ratings', 'discount_price']].dropna()
                
                if len(df_cat_clean) > 10:  # Minimum sample size
                    corr, pvalue = stats.spearmanr(
                        df_cat_clean['no_of_ratings'], 
                        df_cat_clean['discount_price']
                    )
                    
                    category_corrs.append({
                        'category': category,
                        'correlation': round(corr, 3),
                        'p_value': round(pvalue, 4),
                        'sample_size': len(df_cat_clean)
                    })
            
            results['by_category'] = pd.DataFrame(category_corrs).sort_values(
                'correlation', ascending=False
            )
        
        return results
    
    def create_demand_heatmap(self, price_bins: int = 10, 
                             demand_bins: int = 10) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Create demand vs price heatmap
        
        Args:
            price_bins: Number of price bins
            demand_bins: Number of demand bins
        
        Returns:
            tuple: (heatmap_data, figure)
        """
        df = self.df[['discount_price', 'no_of_ratings']].dropna()
        
        # Create bins
        df['price_bin'] = pd.qcut(df['discount_price'], q=price_bins, 
                                   labels=[f'P{i+1}' for i in range(price_bins)], 
                                   duplicates='drop')
        df['demand_bin'] = pd.qcut(df['no_of_ratings'], q=demand_bins, 
                                    labels=[f'D{i+1}' for i in range(demand_bins)], 
                                    duplicates='drop')
        
        # Create pivot table
        heatmap_data = df.groupby(['price_bin', 'demand_bin']).size().unstack(fill_value=0)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('Product Distribution: Price vs Demand', fontsize=14, fontweight='bold')
        ax.set_xlabel('Demand Level (Low to High)', fontsize=12)
        ax.set_ylabel('Price Level (Low to High)', fontsize=12)
        
        plt.tight_layout()
        
        return heatmap_data, fig
    
    def get_demand_summary(self) -> Dict:
        """
        Get overall demand summary statistics
        
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        if 'no_of_ratings' in self.df.columns:
            summary['total_reviews'] = int(self.df['no_of_ratings'].sum())
            summary['avg_reviews_per_product'] = round(self.df['no_of_ratings'].mean(), 1)
            summary['median_reviews'] = int(self.df['no_of_ratings'].median())
            summary['max_reviews'] = int(self.df['no_of_ratings'].max())
            summary['products_with_reviews'] = int((self.df['no_of_ratings'] > 0).sum())
            summary['pct_with_reviews'] = round(
                (self.df['no_of_ratings'] > 0).sum() / len(self.df) * 100, 1
            )
        
        if 'ratings' in self.df.columns:
            summary['avg_rating'] = round(self.df['ratings'].mean(), 2)
            summary['median_rating'] = round(self.df['ratings'].median(), 2)
            summary['products_highly_rated'] = int((self.df['ratings'] >= 4.0).sum())
        
        return summary
    
    def visualize_demand_distribution(self, category_col: str = 'main_category', 
                                     top_n: int = 15, figsize: tuple = (14, 6)):
        """
        Visualize demand distribution across categories
        
        Args:
            category_col: Category column
            top_n: Number of categories to show
            figsize: Figure size
        """
        # Get top categories by total demand
        category_demand = self.df.groupby(category_col)['no_of_ratings'].sum().sort_values(ascending=False)
        top_categories = category_demand.head(top_n).index
        
        df_top = self.df[self.df[category_col].isin(top_categories)].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Total reviews by category (bar chart)
        category_totals = df_top.groupby(category_col)['no_of_ratings'].sum().sort_values()
        category_totals.plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title(f'Total Reviews by Category (Top {top_n})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Total Reviews')
        axes[0].set_ylabel('Category')
        
        # Average reviews per product (bar chart)
        category_avg = df_top.groupby(category_col)['no_of_ratings'].mean().sort_values()
        category_avg.plot(kind='barh', ax=axes[1], color='coral')
        axes[1].set_title('Avg Reviews per Product', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Average Reviews')
        axes[1].set_ylabel('Category')
        
        plt.tight_layout()
        return fig
