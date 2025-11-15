"""
Data loading and cleaning utilities
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, List
from src.utils.config import get_dataset_path, PRICE_COLUMNS, RATING_COLUMNS


def clean_price(price_str):
    """
    Clean price string by removing currency symbols and commas
    
    Args:
        price_str: Price string (e.g., '₹32,999')
    
    Returns:
        float: Cleaned price value
    """
    if pd.isna(price_str):
        return np.nan
    
    # Remove currency symbols, commas, and whitespace
    cleaned = re.sub(r'[₹,\s]', '', str(price_str))
    
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_rating(rating_str):
    """
    Clean rating string and convert to float
    
    Args:
        rating_str: Rating string
    
    Returns:
        float: Cleaned rating value
    """
    if pd.isna(rating_str):
        return np.nan
    
    try:
        return float(str(rating_str).strip())
    except ValueError:
        return np.nan


def clean_number_of_ratings(rating_count_str):
    """
    Clean number of ratings string
    
    Args:
        rating_count_str: Rating count string (e.g., '1,234')
    
    Returns:
        int: Cleaned rating count
    """
    if pd.isna(rating_count_str):
        return np.nan
    
    # Remove commas and whitespace
    cleaned = re.sub(r'[,\s]', '', str(rating_count_str))
    
    try:
        return int(float(cleaned))
    except ValueError:
        return np.nan


def load_and_clean_data(file_path: Optional[str] = None, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load and clean the Amazon products dataset
    
    Args:
        file_path: Path to CSV file. If None, uses the path from dataset_path.txt
        sample_frac: Fraction of data to sample (for testing)
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Get dataset path
    if file_path is None:
        file_path = get_dataset_path()
        if file_path is None:
            raise FileNotFoundError("Dataset path not found. Run download_dataset.py first.")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Sample if requested
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df):,} rows ({sample_frac*100:.1f}%)")
    
    print(f"Initial shape: {df.shape}")
    
    # Clean price columns
    if 'actual_price' in df.columns:
        df['actual_price'] = df['actual_price'].apply(clean_price)
    
    if 'discount_price' in df.columns:
        df['discount_price'] = df['discount_price'].apply(clean_price)
    
    # Clean rating columns
    if 'ratings' in df.columns:
        df['ratings'] = df['ratings'].apply(clean_rating)
    
    if 'no_of_ratings' in df.columns:
        df['no_of_ratings'] = df['no_of_ratings'].apply(clean_number_of_ratings)
    
    # Calculate discount percentage
    if 'actual_price' in df.columns and 'discount_price' in df.columns:
        df['discount_percent'] = np.where(
            (df['actual_price'] > 0) & (df['discount_price'] > 0),
            ((df['actual_price'] - df['discount_price']) / df['actual_price']) * 100,
            np.nan
        )
    
    # Fill missing category columns
    if 'main_category' in df.columns:
        df['main_category'] = df['main_category'].fillna('Unknown')
    
    if 'sub_category' in df.columns:
        df['sub_category'] = df['sub_category'].fillna('Unknown')
    
    print(f"✓ Data cleaned successfully!")
    print(f"Final shape: {df.shape}")
    
    return df


def load_all_categories(base_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load and combine all category CSV files
    
    Args:
        base_path: Base directory containing CSV files
        limit: Maximum number of files to load
    
    Returns:
        pd.DataFrame: Combined dataframe
    """
    base_path = Path(base_path)
    csv_files = list(base_path.glob("*.csv"))
    
    if limit:
        csv_files = csv_files[:limit]
    
    print(f"Loading {len(csv_files)} CSV files...")
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  ✓ Loaded {file.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  ✗ Error loading {file.name}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {len(combined_df):,} rows")
    
    # Clean the combined data
    return clean_data(combined_df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to a dataframe
    
    Args:
        df: Raw dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()
    
    # Clean price columns
    if 'actual_price' in df.columns:
        df['actual_price'] = df['actual_price'].apply(clean_price)
    
    if 'discount_price' in df.columns:
        df['discount_price'] = df['discount_price'].apply(clean_price)
    
    # Clean rating columns
    if 'ratings' in df.columns:
        df['ratings'] = df['ratings'].apply(clean_rating)
    
    if 'no_of_ratings' in df.columns:
        df['no_of_ratings'] = df['no_of_ratings'].apply(clean_number_of_ratings)
    
    # Calculate discount percentage
    if 'actual_price' in df.columns and 'discount_price' in df.columns:
        df['discount_percent'] = np.where(
            (df['actual_price'] > 0) & (df['discount_price'] > 0),
            ((df['actual_price'] - df['discount_price']) / df['actual_price']) * 100,
            np.nan
        )
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset
    
    Args:
        df: Dataframe to summarize
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_products': len(df),
        'categories': df['main_category'].nunique() if 'main_category' in df.columns else 0,
        'subcategories': df['sub_category'].nunique() if 'sub_category' in df.columns else 0,
        'avg_price': df['actual_price'].mean() if 'actual_price' in df.columns else 0,
        'avg_discount': df['discount_percent'].mean() if 'discount_percent' in df.columns else 0,
        'avg_rating': df['ratings'].mean() if 'ratings' in df.columns else 0,
        'total_reviews': df['no_of_ratings'].sum() if 'no_of_ratings' in df.columns else 0,
        'products_with_prices': df['actual_price'].notna().sum() if 'actual_price' in df.columns else 0,
        'products_with_ratings': df['ratings'].notna().sum() if 'ratings' in df.columns else 0,
    }
    
    return summary
