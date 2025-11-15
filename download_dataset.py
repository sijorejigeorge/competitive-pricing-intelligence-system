"""
Download Amazon Products Dataset using kagglehub
"""
import kagglehub
import pandas as pd
import os

# Download latest version
print("Downloading Amazon Products Dataset...")
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")

print(f"✓ Dataset downloaded successfully!")
print(f"Path to dataset files: {path}")

# List all files in the downloaded directory
print("\nFiles in dataset:")
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.2f} MB)")

# Load and explore the dataset
print("\n" + "="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Find the CSV file
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    data_file = os.path.join(path, csv_files[0])
    print(f"\nLoading {csv_files[0]}...")
    
    df = pd.read_csv(data_file)
    
    print(f"\n✓ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    
    print("\n📊 DATASET OVERVIEW")
    print("-" * 80)
    print(df.info())
    
    print("\n📋 COLUMN NAMES")
    print("-" * 80)
    print(df.columns.tolist())
    
    print("\n🔍 FIRST 5 ROWS")
    print("-" * 80)
    print(df.head())
    
    print("\n📈 BASIC STATISTICS")
    print("-" * 80)
    print(df.describe())
    
    print("\n🏷️ UNIQUE CATEGORIES")
    print("-" * 80)
    if 'main_category' in df.columns:
        print(f"Main Categories: {df['main_category'].nunique()}")
        print(df['main_category'].value_counts().head(10))
    
    if 'sub_category' in df.columns:
        print(f"\nSub Categories: {df['sub_category'].nunique()}")
    
    print("\n💰 PRICE RANGE")
    print("-" * 80)
    
    # Clean and convert price columns
    def clean_price(price_str):
        """Convert price string like '₹58,990' to float"""
        if pd.isna(price_str):
            return None
        try:
            # Remove ₹ symbol and commas, then convert to float
            return float(str(price_str).replace('₹', '').replace(',', '').strip())
        except:
            return None
    
    if 'actual_price' in df.columns:
        actual_prices = df['actual_price'].apply(clean_price)
        valid_actual = actual_prices.dropna()
        if len(valid_actual) > 0:
            print(f"Actual Price: ₹{valid_actual.min():.2f} - ₹{valid_actual.max():.2f}")
            print(f"  Average: ₹{valid_actual.mean():.2f}")
    
    if 'discount_price' in df.columns:
        discount_prices = df['discount_price'].apply(clean_price)
        valid_discount = discount_prices.dropna()
        if len(valid_discount) > 0:
            print(f"Discount Price: ₹{valid_discount.min():.2f} - ₹{valid_discount.max():.2f}")
            print(f"  Average: ₹{valid_discount.mean():.2f}")
    
    print("\n⭐ RATINGS OVERVIEW")
    print("-" * 80)
    
    # Clean and convert ratings
    def clean_rating(rating_str):
        """Convert rating string to float"""
        if pd.isna(rating_str):
            return None
        try:
            return float(str(rating_str).strip())
        except:
            return None
    
    def clean_rating_count(count_str):
        """Convert rating count string like '1,234' to int"""
        if pd.isna(count_str):
            return None
        try:
            return int(str(count_str).replace(',', '').strip())
        except:
            return None
    
    if 'ratings' in df.columns:
        ratings = df['ratings'].apply(clean_rating)
        valid_ratings = ratings.dropna()
        if len(valid_ratings) > 0:
            print(f"Average Rating: {valid_ratings.mean():.2f}")
            print(f"Rating Range: {valid_ratings.min():.1f} - {valid_ratings.max():.1f}")
    
    if 'no_of_ratings' in df.columns:
        rating_counts = df['no_of_ratings'].apply(clean_rating_count)
        valid_counts = rating_counts.dropna()
        if len(valid_counts) > 0:
            print(f"Total Reviews: {valid_counts.sum():,}")
            print(f"Avg Reviews per Product: {valid_counts.mean():.0f}")
            print(f"Max Reviews on Single Product: {valid_counts.max():,}")
    
    print("\n❌ MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False))
    
    # Save path for later use
    with open('dataset_path.txt', 'w') as f:
        f.write(data_file)
    
    print(f"\n✓ Dataset path saved to 'dataset_path.txt'")
    print(f"\n{'='*80}")
    print("Dataset is ready for analysis!")
    print(f"{'='*80}")
else:
    print("No CSV file found in the downloaded dataset.")
