"""
Competitive opportunity detection module
Identifies market gaps, price wars, premium segments, and strategic opportunities
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class OpportunityDetector:
    """
    Detect competitive opportunities and market gaps
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize opportunity detector
        
        Args:
            df: Preprocessed dataframe with pricing and demand data
        """
        self.df = df.copy()
        self._add_derived_features()
    
    def _add_derived_features(self):
        """Add features needed for opportunity detection"""
        # Popularity index if not present
        if 'popularity_index' not in self.df.columns and 'no_of_ratings' in self.df.columns:
            self.df['popularity_index'] = np.log10(self.df['no_of_ratings'].fillna(0) + 1)
        
        # Quality score if not present
        if 'quality_score' not in self.df.columns:
            if 'ratings' in self.df.columns and 'popularity_index' in self.df.columns:
                self.df['quality_score'] = (
                    self.df['ratings'].fillna(0) * self.df['popularity_index']
                )
    
    def find_market_gaps(self, min_demand_percentile: float = 60, 
                        max_competition_percentile: float = 40) -> pd.DataFrame:
        """
        Find categories with high demand but low competition
        
        Args:
            min_demand_percentile: Minimum demand percentile
            max_competition_percentile: Maximum competition percentile
        
        Returns:
            pd.DataFrame: Market gap opportunities
        """
        # Calculate category-level metrics
        category_metrics = self.df.groupby('main_category').agg({
            'no_of_ratings': 'sum',  # Total demand
            'name': 'count',  # Number of products (competition)
            'ratings': 'mean',  # Average quality
            'discount_price': 'mean',  # Average price
            'discount_percent': 'mean'  # Average discount
        }).reset_index()
        
        category_metrics.columns = [
            'category', 'total_demand', 'product_count', 
            'avg_rating', 'avg_price', 'avg_discount'
        ]
        
        # Calculate demand per product (demand intensity)
        category_metrics['demand_intensity'] = (
            category_metrics['total_demand'] / category_metrics['product_count']
        )
        
        # Find high demand, low competition
        demand_threshold = category_metrics['total_demand'].quantile(min_demand_percentile / 100)
        competition_threshold = category_metrics['product_count'].quantile(max_competition_percentile / 100)
        
        gaps = category_metrics[
            (category_metrics['total_demand'] >= demand_threshold) &
            (category_metrics['product_count'] <= competition_threshold) &
            (category_metrics['avg_rating'] >= 3.5)  # Ensure quality
        ].copy()
        
        # Calculate opportunity score
        gaps['opportunity_score'] = (
            (gaps['demand_intensity'] / gaps['demand_intensity'].max()) * 50 +
            ((gaps['product_count'].max() - gaps['product_count']) / gaps['product_count'].max()) * 30 +
            (gaps['avg_rating'] / 5) * 20
        ).round(2)
        
        gaps = gaps.sort_values('opportunity_score', ascending=False)
        
        print(f"🎯 Found {len(gaps)} market gap opportunities")
        
        return gaps
    
    def detect_price_wars(self, discount_threshold: float = 30, 
                         demand_threshold: float = 70) -> pd.DataFrame:
        """
        Detect price wars (high discount + high demand = intense competition)
        
        Args:
            discount_threshold: Minimum discount % to consider
            demand_threshold: Demand percentile threshold
        
        Returns:
            pd.DataFrame: Products in price wars
        """
        if 'discount_percent' not in self.df.columns or 'no_of_ratings' not in self.df.columns:
            print("Required columns not found")
            return pd.DataFrame()
        
        demand_cutoff = self.df['no_of_ratings'].quantile(demand_threshold / 100)
        
        price_wars = self.df[
            (self.df['discount_percent'] >= discount_threshold) &
            (self.df['no_of_ratings'] >= demand_cutoff)
        ].copy()
        
        # Calculate competitive intensity
        price_wars['competitive_intensity'] = (
            (price_wars['discount_percent'] / 100) * 
            np.log10(price_wars['no_of_ratings'] + 1)
        ).round(2)
        
        price_wars = price_wars.sort_values('competitive_intensity', ascending=False)
        
        print(f"⚔️ Found {len(price_wars)} products in price wars")
        print(f"   (>{discount_threshold}% discount, >{demand_threshold}th percentile demand)")
        
        return price_wars[['name', 'main_category', 'discount_price', 'actual_price', 
                          'discount_percent', 'ratings', 'no_of_ratings', 'competitive_intensity']]
    
    def identify_premium_segments(self, price_percentile: float = 75, 
                                  rating_threshold: float = 4.2,
                                  demand_percentile: float = 50) -> pd.DataFrame:
        """
        Identify premium segments (high price + high quality + decent demand)
        
        Args:
            price_percentile: Price percentile threshold
            rating_threshold: Minimum rating
            demand_percentile: Minimum demand percentile
        
        Returns:
            pd.DataFrame: Premium segment products
        """
        price_cutoff = self.df['discount_price'].quantile(price_percentile / 100)
        demand_cutoff = self.df['no_of_ratings'].quantile(demand_percentile / 100)
        
        premium = self.df[
            (self.df['discount_price'] >= price_cutoff) &
            (self.df['ratings'] >= rating_threshold) &
            (self.df['no_of_ratings'] >= demand_cutoff)
        ].copy()
        
        # Calculate premium score
        premium['premium_score'] = (
            (premium['discount_price'] / self.df['discount_price'].max()) * 40 +
            (premium['ratings'] / 5) * 40 +
            (np.log10(premium['no_of_ratings'] + 1) / np.log10(self.df['no_of_ratings'].max() + 1)) * 20
        ).round(2)
        
        premium = premium.sort_values('premium_score', ascending=False)
        
        print(f"💎 Found {len(premium)} premium segment products")
        
        return premium[['name', 'main_category', 'discount_price', 'ratings', 
                       'no_of_ratings', 'premium_score']]
    
    def find_underperformers(self, price_percentile: float = 60, 
                            demand_percentile: float = 30,
                            rating_threshold: float = 3.5) -> pd.DataFrame:
        """
        Find underperformers (high price + low demand = overpriced)
        
        Args:
            price_percentile: Price percentile threshold
            demand_percentile: Maximum demand percentile
            rating_threshold: Maximum rating
        
        Returns:
            pd.DataFrame: Underperforming products
        """
        price_cutoff = self.df['discount_price'].quantile(price_percentile / 100)
        demand_cutoff = self.df['no_of_ratings'].quantile(demand_percentile / 100)
        
        underperformers = self.df[
            (self.df['discount_price'] >= price_cutoff) &
            (self.df['no_of_ratings'] <= demand_cutoff) &
            (self.df['ratings'] <= rating_threshold)
        ].copy()
        
        print(f"⚠️ Found {len(underperformers)} underperforming products")
        print(f"   (high price, low demand, mediocre ratings)")
        
        return underperformers[['name', 'main_category', 'discount_price', 
                               'ratings', 'no_of_ratings']]
    
    def find_underpriced_gems(self, price_percentile: float = 40, 
                             rating_threshold: float = 4.3,
                             demand_percentile: float = 70) -> pd.DataFrame:
        """
        Find underpriced high-quality products (potential value plays)
        
        Args:
            price_percentile: Maximum price percentile
            rating_threshold: Minimum rating
            demand_percentile: Minimum demand percentile
        
        Returns:
            pd.DataFrame: Underpriced quality products
        """
        price_cutoff = self.df['discount_price'].quantile(price_percentile / 100)
        demand_cutoff = self.df['no_of_ratings'].quantile(demand_percentile / 100)
        
        gems = self.df[
            (self.df['discount_price'] <= price_cutoff) &
            (self.df['ratings'] >= rating_threshold) &
            (self.df['no_of_ratings'] >= demand_cutoff)
        ].copy()
        
        # Calculate value score (quality / price ratio)
        gems['value_score'] = (
            (gems['ratings'] / 5) * 60 +
            (np.log10(gems['no_of_ratings'] + 1) / np.log10(self.df['no_of_ratings'].max() + 1)) * 40 -
            (gems['discount_price'] / self.df['discount_price'].max()) * 20
        ).round(2)
        
        gems = gems.sort_values('value_score', ascending=False)
        
        print(f"💰 Found {len(gems)} underpriced high-quality products")
        
        return gems[['name', 'main_category', 'discount_price', 'ratings', 
                    'no_of_ratings', 'value_score']]
    
    def create_opportunity_matrix(self) -> pd.DataFrame:
        """
        Create a comprehensive opportunity matrix for all products
        
        Returns:
            pd.DataFrame: Products with opportunity classifications
        """
        df = self.df.copy()
        
        # Calculate percentiles
        price_percentile = df['discount_price'].rank(pct=True) * 100
        demand_percentile = df['no_of_ratings'].rank(pct=True) * 100
        rating_percentile = df['ratings'].rank(pct=True) * 100
        
        # Classify opportunities
        def classify_opportunity(row):
            price_pct = price_percentile.loc[row.name]
            demand_pct = demand_percentile.loc[row.name]
            rating_pct = rating_percentile.loc[row.name]
            
            # High demand + High price + High rating = Premium
            if demand_pct > 70 and price_pct > 70 and rating_pct > 70:
                return 'Premium Segment'
            
            # High demand + High discount = Price War
            elif demand_pct > 70 and row.get('discount_percent', 0) > 30:
                return 'Price War'
            
            # Low price + High rating + High demand = Underpriced Gem
            elif price_pct < 40 and rating_pct > 70 and demand_pct > 60:
                return 'Underpriced Gem'
            
            # High price + Low demand + Low rating = Underperformer
            elif price_pct > 60 and demand_pct < 30 and rating_pct < 50:
                return 'Underperformer'
            
            # High demand + Medium price = Mass Market
            elif demand_pct > 60 and 40 <= price_pct <= 70:
                return 'Mass Market'
            
            # Low demand + Low price = Budget Segment
            elif demand_pct < 40 and price_pct < 40:
                return 'Budget Segment'
            
            else:
                return 'Neutral'
        
        df['opportunity_type'] = df.apply(classify_opportunity, axis=1)
        
        # Summary
        print("\n📊 Opportunity Matrix Summary:")
        print(df['opportunity_type'].value_counts())
        
        return df
    
    def analyze_category_opportunities(self, category_col: str = 'main_category') -> pd.DataFrame:
        """
        Analyze opportunities by category
        
        Args:
            category_col: Category column
        
        Returns:
            pd.DataFrame: Category opportunity analysis
        """
        df = self.create_opportunity_matrix()
        
        # Count opportunities by category
        category_opp = pd.crosstab(
            df[category_col], 
            df['opportunity_type']
        )
        
        # Calculate total products per category
        category_opp['total_products'] = category_opp.sum(axis=1)
        
        # Calculate opportunity percentages
        for col in category_opp.columns:
            if col != 'total_products':
                category_opp[f'{col}_pct'] = (
                    category_opp[col] / category_opp['total_products'] * 100
                ).round(1)
        
        category_opp = category_opp.reset_index()
        
        return category_opp
    
    def visualize_opportunity_landscape(self, figsize: tuple = (14, 8)):
        """
        Visualize the competitive opportunity landscape
        
        Args:
            figsize: Figure size
        
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        df = self.df[['discount_price', 'no_of_ratings', 'ratings']].dropna()
        
        # Sample for performance if too large
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Price vs Demand scatter
        scatter = axes[0, 0].scatter(
            df['discount_price'], 
            df['no_of_ratings'],
            c=df['ratings'],
            cmap='RdYlGn',
            alpha=0.6,
            s=20
        )
        axes[0, 0].set_xlabel('Price (₹)')
        axes[0, 0].set_ylabel('Number of Reviews (Demand Proxy)')
        axes[0, 0].set_title('Price vs Demand (colored by Rating)')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        plt.colorbar(scatter, ax=axes[0, 0], label='Rating')
        
        # 2. Opportunity type distribution
        if 'opportunity_type' in self.df.columns:
            opp_counts = self.df['opportunity_type'].value_counts()
            axes[0, 1].barh(opp_counts.index, opp_counts.values, color='steelblue')
            axes[0, 1].set_xlabel('Number of Products')
            axes[0, 1].set_title('Opportunity Type Distribution')
        
        # 3. Price distribution by quartile
        df['demand_quartile'] = pd.qcut(df['no_of_ratings'], q=4, 
                                         labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        df.boxplot(column='discount_price', by='demand_quartile', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Demand Quartile')
        axes[1, 0].set_ylabel('Price (₹)')
        axes[1, 0].set_title('Price Distribution by Demand Level')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=0)
        
        # 4. Quality-Price matrix
        df['price_tier'] = pd.qcut(df['discount_price'], q=3, 
                                    labels=['Low', 'Medium', 'High'], duplicates='drop')
        df['rating_tier'] = pd.cut(df['ratings'], bins=[0, 3, 4, 5.1], 
                                    labels=['Low', 'Medium', 'High'])
        
        matrix = pd.crosstab(df['price_tier'], df['rating_tier'])
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Product Distribution: Price vs Quality')
        axes[1, 1].set_xlabel('Rating Tier')
        axes[1, 1].set_ylabel('Price Tier')
        
        plt.tight_layout()
        return fig
    
    def get_opportunity_summary(self) -> Dict:
        """
        Get summary of all opportunities
        
        Returns:
            dict: Opportunity summary
        """
        summary = {}
        
        # Market gaps
        gaps = self.find_market_gaps()
        summary['market_gaps'] = len(gaps)
        
        # Price wars
        price_wars = self.detect_price_wars()
        summary['price_war_products'] = len(price_wars)
        
        # Premium segments
        premium = self.identify_premium_segments()
        summary['premium_products'] = len(premium)
        
        # Underpriced gems
        gems = self.find_underpriced_gems()
        summary['underpriced_gems'] = len(gems)
        
        # Underperformers
        underperf = self.find_underperformers()
        summary['underperformers'] = len(underperf)
        
        return summary
