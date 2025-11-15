"""
Product clustering and competitive benchmarking using NLP
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ProductClusterer:
    """
    Cluster similar products for competitive benchmarking
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize product clusterer
        
        Args:
            df: Preprocessed dataframe with product data
        """
        self.df = df.copy()
        self.vectorizer = None
        self.clusters = None
    
    def extract_text_features(self, text_col: str = 'name', 
                             max_features: int = 100) -> np.ndarray:
        """
        Extract TF-IDF features from product names
        
        Args:
            text_col: Column containing text data
            max_features: Maximum number of features
        
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if text_col not in self.df.columns:
            print(f"Column {text_col} not found")
            return np.array([])
        
        # Clean text data
        texts = self.df[text_col].fillna('').astype(str)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8  # Maximum document frequency
        )
        
        # Fit and transform
        features = self.vectorizer.fit_transform(texts)
        
        print(f"✓ Extracted {features.shape[1]} text features from {len(texts)} products")
        
        return features.toarray()
    
    def cluster_products(self, n_clusters: int = 10, 
                        method: str = 'kmeans',
                        text_col: str = 'name',
                        max_features: int = 100) -> pd.DataFrame:
        """
        Cluster products based on text similarity
        
        Args:
            n_clusters: Number of clusters (for KMeans)
            method: 'kmeans' or 'dbscan'
            text_col: Column with product text
            max_features: Maximum TF-IDF features
        
        Returns:
            pd.DataFrame: Dataframe with cluster assignments
        """
        # Extract features
        features = self.extract_text_features(text_col, max_features)
        
        if features.size == 0:
            return self.df
        
        # Cluster
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features)
        
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(features)
        
        else:
            print(f"Unknown method: {method}")
            return self.df
        
        # Add cluster labels to dataframe
        df_clustered = self.df.copy()
        df_clustered['cluster'] = cluster_labels
        
        self.clusters = df_clustered
        
        # Print summary
        print(f"\n✓ Clustered into {df_clustered['cluster'].nunique()} groups")
        print(f"Cluster sizes: {df_clustered['cluster'].value_counts().head(10).to_dict()}")
        
        return df_clustered
    
    def get_cluster_summary(self, cluster_id: int) -> Dict:
        """
        Get summary statistics for a specific cluster
        
        Args:
            cluster_id: Cluster ID
        
        Returns:
            dict: Cluster summary
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return {}
        
        cluster_data = self.clusters[self.clusters['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            return {'error': 'Cluster not found'}
        
        summary = {
            'cluster_id': cluster_id,
            'product_count': len(cluster_data),
            'avg_price': cluster_data['discount_price'].mean(),
            'median_price': cluster_data['discount_price'].median(),
            'price_range': (cluster_data['discount_price'].min(), cluster_data['discount_price'].max()),
            'avg_rating': cluster_data['ratings'].mean(),
            'avg_reviews': cluster_data['no_of_ratings'].mean(),
            'top_categories': cluster_data['main_category'].value_counts().head(3).to_dict(),
            'sample_products': cluster_data['name'].head(5).tolist()
        }
        
        return summary
    
    def compare_within_cluster(self, cluster_id: int, 
                               sort_by: str = 'discount_price') -> pd.DataFrame:
        """
        Compare products within the same cluster (competitive benchmarking)
        
        Args:
            cluster_id: Cluster ID
            sort_by: Column to sort by
        
        Returns:
            pd.DataFrame: Ranked products in cluster
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return pd.DataFrame()
        
        cluster_data = self.clusters[self.clusters['cluster'] == cluster_id].copy()
        
        if len(cluster_data) == 0:
            return pd.DataFrame()
        
        # Calculate competitive metrics
        cluster_data['price_rank'] = cluster_data['discount_price'].rank()
        cluster_data['quality_rank'] = cluster_data['ratings'].rank(ascending=False)
        cluster_data['demand_rank'] = cluster_data['no_of_ratings'].rank(ascending=False)
        
        # Calculate competitive score (lower is better)
        cluster_data['competitive_score'] = (
            cluster_data['price_rank'] * 0.3 +  # Lower price is better
            cluster_data['quality_rank'] * 0.4 +  # Higher quality is better
            cluster_data['demand_rank'] * 0.3  # Higher demand is better
        )
        
        # Sort
        cluster_data = cluster_data.sort_values('competitive_score')
        
        return cluster_data[['name', 'discount_price', 'ratings', 'no_of_ratings', 
                            'price_rank', 'quality_rank', 'demand_rank', 'competitive_score']]
    
    def find_similar_products(self, product_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Find products similar to a given product
        
        Args:
            product_name: Product name to search for
            top_n: Number of similar products to return
        
        Returns:
            pd.DataFrame: Similar products
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return pd.DataFrame()
        
        # Find the product
        matches = self.clusters[self.clusters['name'].str.contains(product_name, case=False, na=False)]
        
        if len(matches) == 0:
            print(f"Product not found: {product_name}")
            return pd.DataFrame()
        
        # Get cluster of first match
        target_cluster = matches.iloc[0]['cluster']
        
        # Get products in same cluster
        similar = self.clusters[self.clusters['cluster'] == target_cluster].copy()
        
        # Exclude the target product
        similar = similar[~similar['name'].str.contains(product_name, case=False, na=False)]
        
        # Sort by similarity (using price and ratings as proxy)
        target_price = matches.iloc[0]['discount_price']
        target_rating = matches.iloc[0]['ratings']
        
        similar['similarity_score'] = (
            1 / (1 + abs(similar['discount_price'] - target_price) / target_price) * 0.5 +
            1 / (1 + abs(similar['ratings'] - target_rating)) * 0.5
        )
        
        similar = similar.sort_values('similarity_score', ascending=False).head(top_n)
        
        return similar[['name', 'discount_price', 'ratings', 'no_of_ratings', 'similarity_score']]
    
    def analyze_cluster_characteristics(self) -> pd.DataFrame:
        """
        Analyze characteristics of all clusters
        
        Returns:
            pd.DataFrame: Cluster characteristics
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return pd.DataFrame()
        
        cluster_stats = self.clusters.groupby('cluster').agg({
            'name': 'count',
            'discount_price': ['mean', 'std'],
            'ratings': 'mean',
            'no_of_ratings': ['mean', 'sum'],
            'discount_percent': 'mean'
        }).round(2)
        
        # Flatten columns
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        cluster_stats = cluster_stats.reset_index()
        
        # Rename
        cluster_stats.columns = [
            'cluster_id', 'product_count', 
            'avg_price', 'price_std',
            'avg_rating', 
            'avg_reviews', 'total_reviews',
            'avg_discount'
        ]
        
        # Calculate cluster competitiveness
        cluster_stats['competitiveness'] = (
            cluster_stats['avg_discount'] / 100 * 
            np.log10(cluster_stats['total_reviews'] + 1)
        ).round(2)
        
        # Categorize clusters
        def categorize_cluster(row):
            if row['avg_price'] > cluster_stats['avg_price'].median():
                if row['avg_rating'] > 4.0:
                    return 'Premium'
                else:
                    return 'Overpriced'
            else:
                if row['avg_reviews'] > cluster_stats['avg_reviews'].median():
                    return 'Value'
                else:
                    return 'Budget'
        
        cluster_stats['category'] = cluster_stats.apply(categorize_cluster, axis=1)
        
        return cluster_stats.sort_values('product_count', ascending=False)
    
    def get_top_products_per_cluster(self, metric: str = 'competitive_score', 
                                    top_n: int = 3) -> pd.DataFrame:
        """
        Get top products from each cluster
        
        Args:
            metric: Metric to rank by
            top_n: Number of products per cluster
        
        Returns:
            pd.DataFrame: Top products per cluster
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return pd.DataFrame()
        
        top_products = []
        
        for cluster_id in self.clusters['cluster'].unique():
            cluster_comparison = self.compare_within_cluster(cluster_id)
            
            if not cluster_comparison.empty:
                top_in_cluster = cluster_comparison.head(top_n)
                top_in_cluster['cluster'] = cluster_id
                top_products.append(top_in_cluster)
        
        if top_products:
            result = pd.concat(top_products, ignore_index=True)
            return result
        
        return pd.DataFrame()
    
    def visualize_clusters(self, max_clusters: int = 20, figsize: tuple = (14, 10)):
        """
        Visualize product clusters
        
        Args:
            max_clusters: Maximum clusters to visualize
            figsize: Figure size
        
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return None
        
        # Sample data if too large
        df_viz = self.clusters.copy()
        if len(df_viz) > 2000:
            df_viz = df_viz.sample(2000, random_state=42)
        
        # Extract features again for PCA
        features = self.extract_text_features('name', max_features=50)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features[:len(df_viz)])
        
        df_viz['pc1'] = features_2d[:, 0]
        df_viz['pc2'] = features_2d[:, 1]
        
        # Limit clusters for visualization
        top_clusters = df_viz['cluster'].value_counts().head(max_clusters).index
        df_viz_filtered = df_viz[df_viz['cluster'].isin(top_clusters)]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cluster scatter plot
        scatter = axes[0, 0].scatter(
            df_viz_filtered['pc1'], 
            df_viz_filtered['pc2'],
            c=df_viz_filtered['cluster'],
            cmap='tab20',
            alpha=0.6,
            s=30
        )
        axes[0, 0].set_xlabel('Principal Component 1')
        axes[0, 0].set_ylabel('Principal Component 2')
        axes[0, 0].set_title('Product Clusters (PCA Visualization)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # 2. Cluster sizes
        cluster_sizes = df_viz['cluster'].value_counts().head(15)
        axes[0, 1].barh(range(len(cluster_sizes)), cluster_sizes.values, color='steelblue')
        axes[0, 1].set_yticks(range(len(cluster_sizes)))
        axes[0, 1].set_yticklabels([f'Cluster {c}' for c in cluster_sizes.index])
        axes[0, 1].set_xlabel('Number of Products')
        axes[0, 1].set_title('Cluster Sizes (Top 15)')
        
        # 3. Avg price by cluster
        cluster_prices = df_viz.groupby('cluster')['discount_price'].mean().sort_values(ascending=False).head(15)
        axes[1, 0].barh(range(len(cluster_prices)), cluster_prices.values, color='coral')
        axes[1, 0].set_yticks(range(len(cluster_prices)))
        axes[1, 0].set_yticklabels([f'Cluster {c}' for c in cluster_prices.index])
        axes[1, 0].set_xlabel('Average Price (₹)')
        axes[1, 0].set_title('Avg Price by Cluster (Top 15)')
        
        # 4. Avg rating by cluster
        cluster_ratings = df_viz.groupby('cluster')['ratings'].mean().sort_values(ascending=False).head(15)
        axes[1, 1].barh(range(len(cluster_ratings)), cluster_ratings.values, color='green', alpha=0.7)
        axes[1, 1].set_yticks(range(len(cluster_ratings)))
        axes[1, 1].set_yticklabels([f'Cluster {c}' for c in cluster_ratings.index])
        axes[1, 1].set_xlabel('Average Rating')
        axes[1, 1].set_title('Avg Rating by Cluster (Top 15)')
        axes[1, 1].set_xlim(0, 5)
        
        plt.tight_layout()
        return fig
    
    def export_cluster_report(self, output_file: str = 'cluster_report.csv'):
        """
        Export detailed cluster report
        
        Args:
            output_file: Output CSV file path
        """
        if self.clusters is None:
            print("Run cluster_products() first")
            return
        
        # Get cluster characteristics
        cluster_chars = self.analyze_cluster_characteristics()
        
        # Save
        cluster_chars.to_csv(output_file, index=False)
        print(f"✓ Cluster report saved to {output_file}")
