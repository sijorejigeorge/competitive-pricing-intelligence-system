"""
Pricing competitiveness analysis module
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class PricingAnalyzer:
    """
    Analyze pricing strategies and competitiveness across categories and brands
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize pricing analyzer
        
        Args:
            df: Preprocessed dataframe with pricing data
        """
        self.df = df.copy().reset_index(drop=True)
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist"""
        required_cols = ['discount_price', 'actual_price']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def analyze_by_category(self, category_col: str = 'main_category') -> pd.DataFrame:
        """
        Analyze pricing metrics by category
        
        Args:
            category_col: Column name for category grouping
        
        Returns:
            pd.DataFrame: Category-level pricing insights
        """
        # Group by category
        category_stats = self.df.groupby(category_col).agg({
            'discount_price': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'actual_price': ['mean', 'median'],
            'discount_percent': ['mean', 'median', 'max'],
            'ratings': 'mean',
            'no_of_ratings': 'sum'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.reset_index()
        
        # Rename for clarity
        category_stats.columns = [
            'category', 
            'avg_price', 'median_price', 'price_std', 'min_price', 'max_price', 'product_count',
            'avg_actual_price', 'median_actual_price',
            'avg_discount', 'median_discount', 'max_discount',
            'avg_rating', 'total_reviews'
        ]
        
        # Calculate price range
        category_stats['price_range'] = category_stats['max_price'] - category_stats['min_price']
        
        # Calculate competitiveness index (high discount + high reviews = competitive)
        category_stats['competitiveness_index'] = (
            category_stats['avg_discount'] / 100 * 
            np.log10(category_stats['total_reviews'].fillna(0) + 1)
        ).round(2)
        
        # Sort by product count
        category_stats = category_stats.sort_values('product_count', ascending=False)
        
        return category_stats
    
    def analyze_by_brand(self, brand_col: str = 'brand', min_products: int = 10) -> pd.DataFrame:
        """
        Analyze pricing strategies by brand
        
        Args:
            brand_col: Column name for brand
            min_products: Minimum number of products for a brand to be included
        
        Returns:
            pd.DataFrame: Brand-level pricing insights
        """
        if brand_col not in self.df.columns:
            print(f"Column {brand_col} not found in dataframe")
            return pd.DataFrame()
        
        # Group by brand
        brand_stats = self.df.groupby(brand_col).agg({
            'discount_price': ['mean', 'median', 'count'],
            'actual_price': 'mean',
            'discount_percent': 'mean',
            'ratings': 'mean',
            'no_of_ratings': ['sum', 'mean']
        }).round(2)
        
        # Flatten columns
        brand_stats.columns = ['_'.join(col).strip() for col in brand_stats.columns]
        brand_stats = brand_stats.reset_index()
        
        # Rename
        brand_stats.columns = [
            'brand', 
            'avg_price', 'median_price', 'product_count',
            'avg_actual_price', 'avg_discount',
            'avg_rating', 'total_reviews', 'avg_reviews_per_product'
        ]
        
        # Filter by minimum products
        brand_stats = brand_stats[brand_stats['product_count'] >= min_products]
        
        # Calculate brand premium (vs market average)
        market_avg_price = self.df['discount_price'].mean()
        brand_stats['price_premium_vs_market'] = (
            ((brand_stats['avg_price'] - market_avg_price) / market_avg_price) * 100
        ).round(2)
        
        # Sort by product count
        brand_stats = brand_stats.sort_values('product_count', ascending=False)
        
        return brand_stats
    
    def find_pricing_outliers(self, method: str = 'iqr', factor: float = 3.0) -> pd.DataFrame:
        """
        Identify products with unusual pricing
        
        Args:
            method: 'iqr' or 'zscore'
            factor: Threshold factor
        
        Returns:
            pd.DataFrame: Outlier products
        """
        df = self.df.copy()
        
        if method == 'iqr':
            Q1 = df['discount_price'].quantile(0.25)
            Q3 = df['discount_price'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = df[
                (df['discount_price'] < lower_bound) | 
                (df['discount_price'] > upper_bound)
            ].copy()
        
        elif method == 'zscore':
            from scipy import stats
            # Filter out NaN prices using loc to maintain index alignment
            mask = df['discount_price'].notna()
            df_valid = df.loc[mask].copy().reset_index(drop=True)
            z_scores = np.abs(stats.zscore(df_valid['discount_price']))
            outliers = df_valid.loc[z_scores > factor].copy()
        
        else:
            print(f"Unknown method: {method}")
            return pd.DataFrame()
        
        # Add outlier type
        median_price = df['discount_price'].median()
        outliers['outlier_type'] = np.where(
            outliers['discount_price'] > median_price,
            'Overpriced',
            'Underpriced'
        )
        
        return outliers[['name', 'discount_price', 'actual_price', 'discount_percent', 
                        'ratings', 'no_of_ratings', 'outlier_type']]
    
    def compare_discount_strategies(self, category_col: str = 'main_category', 
                                   top_n: int = 20) -> pd.DataFrame:
        """
        Compare discount strategies across categories
        
        Args:
            category_col: Category column
            top_n: Number of top categories to analyze
        
        Returns:
            pd.DataFrame: Discount strategy comparison
        """
        # Get top categories by product count
        top_categories = self.df[category_col].value_counts().head(top_n).index
        df_top = self.df[self.df[category_col].isin(top_categories)].copy()
        
        # Analyze discount patterns
        discount_stats = df_top.groupby(category_col).agg({
            'discount_percent': ['mean', 'median', 'std', 'min', 'max'],
            'discount_price': 'mean',
            'actual_price': 'mean'
        }).round(2)
        
        # Flatten columns
        discount_stats.columns = ['_'.join(col).strip() for col in discount_stats.columns]
        discount_stats = discount_stats.reset_index()
        
        # Calculate discount aggressiveness
        discount_stats['discount_aggressiveness'] = (
            discount_stats['discount_percent_mean'] * 
            (1 + discount_stats['discount_percent_std'] / 100)
        ).round(2)
        
        # Sort by avg discount
        discount_stats = discount_stats.sort_values('discount_percent_mean', ascending=False)
        
        return discount_stats
    
    def calculate_price_premium(self, group_by: str = 'brand', 
                                reference: str = 'category') -> pd.DataFrame:
        """
        Calculate price premium for brands vs category average
        
        Args:
            group_by: Column to group by (usually 'brand')
            reference: Reference level ('category' or 'market')
        
        Returns:
            pd.DataFrame: Price premium analysis
        """
        if group_by not in self.df.columns:
            print(f"Column {group_by} not found")
            return pd.DataFrame()
        
        df = self.df.copy()
        
        # Calculate reference prices
        if reference == 'category' and 'main_category' in df.columns:
            # Category average prices
            category_avg = df.groupby('main_category')['discount_price'].mean()
            df['reference_price'] = df['main_category'].map(category_avg)
        else:
            # Market average
            df['reference_price'] = df['discount_price'].mean()
        
        # Calculate premium
        premium = df.groupby(group_by).agg({
            'discount_price': 'mean',
            'reference_price': 'first',
            'name': 'count'
        }).round(2)
        
        premium.columns = ['avg_price', 'reference_price', 'product_count']
        premium = premium.reset_index()
        
        # Calculate premium percentage
        premium['price_premium_pct'] = (
            ((premium['avg_price'] - premium['reference_price']) / 
             premium['reference_price']) * 100
        ).round(2)
        
        # Categorize
        premium['positioning'] = pd.cut(
            premium['price_premium_pct'],
            bins=[-float('inf'), -20, -5, 5, 20, float('inf')],
            labels=['Budget', 'Value', 'Market Average', 'Premium', 'Luxury']
        )
        
        premium = premium.sort_values('price_premium_pct', ascending=False)
        
        return premium
    
    def get_pricing_summary(self) -> Dict:
        """
        Get overall pricing summary statistics
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_products': len(self.df),
            'avg_actual_price': self.df['actual_price'].mean(),
            'median_actual_price': self.df['actual_price'].median(),
            'avg_discount_price': self.df['discount_price'].mean(),
            'median_discount_price': self.df['discount_price'].median(),
            'avg_discount_percent': self.df['discount_percent'].mean(),
            'max_discount_percent': self.df['discount_percent'].max(),
            'price_range': f"₹{self.df['discount_price'].min():.0f} - ₹{self.df['discount_price'].max():.0f}",
            'products_with_discount': (self.df['discount_percent'] > 0).sum(),
            'products_on_heavy_discount': (self.df['discount_percent'] > 30).sum(),
        }
        
        return summary
    
    def visualize_price_distribution(self, category_col: str = 'main_category', 
                                     top_n: int = 10, figsize: tuple = (14, 6)):
        """
        Visualize price distribution across categories
        
        Args:
            category_col: Category column
            top_n: Number of categories to show
            figsize: Figure size
        """
        # Get top categories
        top_categories = self.df[category_col].value_counts().head(top_n).index
        df_top = self.df[self.df[category_col].isin(top_categories)].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        df_top.boxplot(column='discount_price', by=category_col, ax=axes[0])
        axes[0].set_title('Price Distribution by Category')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Price (₹)')
        plt.sca(axes[0])
        plt.xticks(rotation=45, ha='right')
        
        # Discount distribution
        df_top.boxplot(column='discount_percent', by=category_col, ax=axes[1])
        axes[1].set_title('Discount % by Category')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Discount %')
        plt.sca(axes[1])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
