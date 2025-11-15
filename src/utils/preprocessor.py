"""
Data preprocessing utilities including brand extraction and feature engineering
"""
import pandas as pd
import numpy as np
import re
from typing import List, Optional


class DataPreprocessor:
    """
    Preprocess data for competitive analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with dataframe
        
        Args:
            df: Cleaned dataframe
        """
        self.df = df.copy()
    
    def extract_brands(self, product_name_col: str = 'name') -> pd.Series:
        """
        Extract brand names from product names
        Assumes brand is typically the first word/words before specific product details
        
        Args:
            product_name_col: Column containing product names
        
        Returns:
            pd.Series: Extracted brand names
        """
        def extract_brand(name):
            if pd.isna(name):
                return 'Unknown'
            
            name = str(name).strip()
            
            # Common patterns: Brand name is usually first 1-2 words
            # Split by common delimiters
            words = re.split(r'[\s\-,\(\)]', name)
            words = [w for w in words if w]  # Remove empty
            
            if not words:
                return 'Unknown'
            
            # Take first word as brand
            brand = words[0]
            
            # If first word is very short (< 2 chars), try combining with second
            if len(brand) < 2 and len(words) > 1:
                brand = words[0] + ' ' + words[1]
            
            # Clean brand name
            brand = brand.strip().title()
            
            return brand
        
        brands = self.df[product_name_col].apply(extract_brand)
        return brands
    
    def create_product_features(self) -> pd.DataFrame:
        """
        Create additional features for analysis
        
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        df = self.df.copy()
        
        # Extract brand
        df['brand'] = self.extract_brands()
        
        # Popularity index (log scale for review count)
        if 'no_of_ratings' in df.columns:
            df['popularity_index'] = np.log10(df['no_of_ratings'].fillna(0) + 1)
        
        # Quality score (rating × popularity)
        if 'ratings' in df.columns and 'popularity_index' in df.columns:
            df['quality_score'] = df['ratings'].fillna(0) * df['popularity_index']
        
        # Price tier
        if 'discount_price' in df.columns:
            df['price_tier'] = pd.cut(
                df['discount_price'],
                bins=[0, 500, 1000, 2500, 5000, 10000, float('inf')],
                labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury', 'Ultra-Luxury']
            )
        
        # Discount tier
        if 'discount_percent' in df.columns:
            df['discount_tier'] = pd.cut(
                df['discount_percent'],
                bins=[-1, 0, 10, 20, 30, 50, 100],
                labels=['No Discount', 'Low (1-10%)', 'Medium (11-20%)', 
                       'High (21-30%)', 'Very High (31-50%)', 'Extreme (>50%)']
            )
        
        # Rating category
        if 'ratings' in df.columns:
            df['rating_category'] = pd.cut(
                df['ratings'],
                bins=[0, 2, 3, 4, 4.5, 5.1],
                labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
            )
        
        # Demand category (based on review count)
        if 'no_of_ratings' in df.columns:
            df['demand_category'] = pd.cut(
                df['no_of_ratings'],
                bins=[-1, 0, 10, 100, 1000, 10000, float('inf')],
                labels=['No Demand', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Value score (quality / price ratio)
        if 'quality_score' in df.columns and 'discount_price' in df.columns:
            df['value_score'] = np.where(
                df['discount_price'] > 0,
                df['quality_score'] / (df['discount_price'] / 1000),  # Normalize price
                np.nan
            )
        
        # Revenue potential proxy (price × demand)
        if 'discount_price' in df.columns and 'no_of_ratings' in df.columns:
            df['revenue_potential'] = (
                df['discount_price'].fillna(0) * 
                np.log10(df['no_of_ratings'].fillna(0) + 1)
            )
        
        # Competitive position score
        # High rating + low price = very competitive
        # High price + low rating = weak position
        if 'ratings' in df.columns and 'discount_percent' in df.columns:
            df['competitiveness_score'] = (
                df['ratings'].fillna(0) * 
                (1 + df['discount_percent'].fillna(0) / 100)
            )
        
        return df
    
    def remove_outliers(self, column: str, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from a column
        
        Args:
            column: Column name
            method: 'iqr' or 'zscore'
            factor: IQR multiplier or z-score threshold
        
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        df = self.df.copy()
        
        if column not in df.columns:
            print(f"Column {column} not found")
            return df
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            mask = z_scores < factor
        
        else:
            print(f"Unknown method: {method}")
            return df
        
        removed = len(df) - mask.sum()
        print(f"Removed {removed:,} outliers from {column} ({removed/len(df)*100:.1f}%)")
        
        return df[mask]
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get fully processed dataframe with all features
        
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df = self.create_product_features()
        return df.reset_index(drop=True)


def create_category_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create higher-level category segments
    
    Args:
        df: Dataframe with category columns
    
    Returns:
        pd.DataFrame: Dataframe with segment column
    """
    from src.utils.config import CATEGORY_GROUPS
    
    df = df.copy()
    
    def assign_segment(category):
        if pd.isna(category):
            return 'Other'
        
        category_lower = str(category).lower()
        
        for segment, keywords in CATEGORY_GROUPS.items():
            if any(keyword in category_lower for keyword in keywords):
                return segment
        
        return 'Other'
    
    if 'main_category' in df.columns:
        df['category_segment'] = df['main_category'].apply(assign_segment)
    
    return df
