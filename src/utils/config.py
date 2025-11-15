"""
Configuration file for the Competitive Intelligence System
"""
import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

# Dataset Path
DATASET_PATH_FILE = PROJECT_ROOT / "dataset_path.txt"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data Processing Config
PRICE_COLUMNS = ['actual_price', 'discount_price']
RATING_COLUMNS = ['ratings', 'no_of_ratings']
TEXT_COLUMNS = ['name', 'main_category', 'sub_category']

# Analysis Parameters
MIN_REVIEWS_FOR_DEMAND = 10  # Minimum reviews to consider for demand analysis
POPULARITY_LOG_BASE = 10      # Base for logarithmic popularity index
OUTLIER_THRESHOLD = 3         # Standard deviations for outlier detection

# Clustering Parameters
MAX_CLUSTERS = 10
MIN_CLUSTER_SIZE = 5

# Visualization Config
FIGURE_SIZE = (12, 6)
COLOR_PALETTE = 'viridis'
DPI = 100

# Dashboard Config
DASHBOARD_TITLE = "🏆 Competitive Pricing Intelligence System"
PAGE_ICON = "📊"
LAYOUT = "wide"

# Brand Extraction Keywords (common patterns to clean)
BRAND_KEYWORDS_TO_REMOVE = [
    'Pack of', 'Set of', 'Combo', 'Pair of', 
    'with', 'and', 'for', '&', '-'
]

# Category Groups (for higher-level analysis)
CATEGORY_GROUPS = {
    'Electronics': ['electronics', 'computer', 'mobile', 'laptop', 'camera', 'headphone', 'speaker'],
    'Fashion': ['clothing', 'shoes', 'watch', 'jewel', 'fashion', 'ethnic', 'innerwear'],
    'Home': ['furniture', 'home', 'kitchen', 'bedroom', 'storage', 'lighting'],
    'Appliances': ['appliances', 'refrigerator', 'washing', 'air conditioner'],
    'Sports': ['sports', 'fitness', 'exercise', 'yoga', 'running', 'cycling'],
    'Beauty': ['beauty', 'grooming', 'makeup', 'personal care'],
    'Baby': ['baby', 'kids', 'children'],
    'Automotive': ['car', 'bike', 'motor', 'automotive'],
}

def get_dataset_path():
    """Get the path to the downloaded dataset"""
    if DATASET_PATH_FILE.exists():
        with open(DATASET_PATH_FILE, 'r') as f:
            return f.read().strip()
    return None

def get_processed_data_path():
    """Get path for processed data"""
    return DATA_DIR / "processed_data.csv"

def get_output_path(filename):
    """Get path for output files"""
    return OUTPUT_DIR / filename
