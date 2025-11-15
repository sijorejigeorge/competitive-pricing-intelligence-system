# Competitive Pricing Intelligence System

> A comprehensive competitive analysis platform built on **300K+ Amazon products** across **142 categories**, revealing pricing strategies, market opportunities, and demand patterns.

## Project Overview

This system analyzes competitive pricing dynamics, market positioning, and demand signals using real-world e-commerce data. It provides actionable insights for:
- **Pricing competitiveness** across categories and brands
- **Discount strategies** and promotional patterns
- **Demand estimation** using ratings as a proxy
- **Market gaps** and opportunity detection
- **Price elasticity modeling** for revenue optimization

---

## Key Features

### 1. **Pricing Intelligence**
- Category-level price benchmarking
- Brand premium analysis
- Discount pattern detection
- Pricing outlier identification

### 2. **Demand Analytics**
- Popularity index using review volume
- Quality score combining ratings × demand
- High-demand product detection
- Category saturation analysis

### 3. **Competitive Opportunity Detection**
- Price wars identification (high-demand + low price)
- Premium segments (high-demand + high price)
- Underperformer detection (low-demand + high price)
- Market gap analysis

### 4. **Price Elasticity Modeling**
- Demand-price relationship estimation
- Revenue optimization recommendations
- Category-specific elasticity analysis

### 5. **Product Clustering & Benchmarking**
- NLP-based similar product grouping
- Competitive positioning within clusters
- Brand comparison across similar products

### 6. **Interactive Dashboard**
- Multi-page Streamlit application
- Real-time filtering and exploration
- Exportable insights and visualizations

---

## Project Structure

```
competitive-pricing-intelligence-system/
│
├── data/                          # Processed datasets (empty - excluded from git)
├── notebooks/                     # Jupyter notebooks for exploration
├── outputs/                       # Generated reports and visualizations
│
├── src/                           # Source code
│   ├── utils/                     # Helper functions
│   │   ├── data_loader.py         # Load and clean data
│   │   ├── preprocessor.py        # Data preprocessing
│   │   └── config.py              # Configuration
│   │
│   ├── analysis/                  # Analysis modules
│   │   ├── pricing.py             # Pricing analysis
│   │   ├── demand.py              # Demand modeling
│   │   ├── clustering.py          # Product clustering
│   │   └── opportunities.py       # Opportunity detection
│   │
│   └── models/                    # Statistical models
│       └── elasticity.py          # Price elasticity estimation
│
├── dashboard/                     # Streamlit dashboard
│   ├── app.py                     # Main dashboard
│   └── pages/                     # Dashboard pages
│       ├── 1_Pricing_Intelligence.py
│       ├── 2_Demand_Analysis.py
│       ├── 3_Market_Opportunities.py
│       └── 4_Price_Elasticity.py
│
├── download_dataset.py            # Dataset downloader
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## Quick Start

### 1. **Clone and Setup**
```bash
cd "Pricing and market analysis system"
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. **Download Dataset**
```bash
python download_dataset.py
```

### 3. **Run Analysis**
```bash
# Run the interactive dashboard
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

---

## Dataset Overview

- **Source:** [Amazon Products Sales Dataset 2023 (Kaggle)](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data)
- **Size:** 300K+ products, 179 MB
- **Categories:** 142 main categories
- **Features:**
  - Product name, category, sub-category
  - Ratings (1-5 stars)
  - Number of ratings (demand proxy)
  - Actual price & discount price
  - Product images and links

### Data Quality
- ~40% products have ratings
- ~65% have pricing data
- Prices range from ₹199 to ₹1,49,000

---

## Analysis Modules

### 1. Pricing Competitiveness Analysis
```python
from src.analysis.pricing import PricingAnalyzer

analyzer = PricingAnalyzer(df)
category_insights = analyzer.analyze_by_category()
brand_premium = analyzer.calculate_brand_premium()
```

### 2. Demand Modeling
```python
from src.analysis.demand import DemandModeler

modeler = DemandModeler(df)
popularity_scores = modeler.calculate_popularity_index()
high_demand_products = modeler.identify_high_demand()
```

### 3. Opportunity Detection
```python
from src.analysis.opportunities import OpportunityDetector

detector = OpportunityDetector(df)
opportunities = detector.find_market_gaps()
price_wars = detector.detect_price_wars()
```

### 4. Price Elasticity
```python
from src.models.elasticity import ElasticityModel

model = ElasticityModel(df)
elasticity = model.estimate_price_elasticity()
optimal_price = model.find_revenue_maximizing_price(base_price=1000, elasticity=-1.5)
```

---

## Dashboard

Launch the interactive dashboard:
```bash
streamlit run dashboard/app.py
```

**Pages:**
1. **Pricing Intelligence** - Category pricing, discounts, brand premiums
2. **Demand Analysis** - Popularity trends, quality scores, demand heatmaps
3. **Opportunity Map** - Market gaps, underpriced/overpriced products
4. **Elasticity Insights** - Price-demand curves, revenue optimization

---

## Key Insights & Business Value

### Competitive Intelligence
- Identify **price wars** in electronics (avg 32% discount)
- Detect **premium segments** (fashion jewelry, luxury beauty)
- Find **undervalued categories** with high demand

### Pricing Strategy
- **Optimal pricing** recommendations using elasticity models
- **Discount effectiveness** analysis by category
- **Revenue maximization** strategies

### Market Opportunities
- **Gap analysis**: High-demand, low-supply subcategories
- **Entry points**: Categories with low competition
- **Growth signals**: Rising demand indicators

---

## Technologies Used

- **Data Processing:** pandas, NumPy
- **Statistical Analysis:** SciPy, statsmodels
- **Clustering:** scikit-learn (KMeans)
- **NLP:** NLTK (TF-IDF for product similarity)
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Dashboard:** Streamlit
- **Data Download:** kagglehub

---

## Future Enhancements

### Data & Analysis
- [ ] Jupyter notebooks for exploratory data analysis
- [ ] Time-series forecasting for seasonal trends (requires historical data)
- [ ] Customer segmentation using RFM analysis (requires customer data)
- [ ] Sentiment analysis from product reviews (requires review text data)

### Advanced Features
- [ ] Real-time competitor price tracking API
- [ ] Multi-platform price comparison (Amazon, Flipkart, etc.)
- [ ] Automated pricing recommendation engine
- [ ] A/B testing simulator for pricing strategies
- [ ] Profitability analysis (requires cost data)

### Machine Learning
- [ ] Predictive models for demand forecasting
- [ ] Dynamic pricing algorithms
- [ ] Churn prediction models (requires customer behavior data)
- [ ] Product recommendation system

---

## Use Cases

- **E-commerce Teams:** Competitive pricing strategies
- **Product Managers:** Market positioning insights
- **Data Analysts:** Portfolio for demonstrating skills
- **Business Strategists:** Market opportunity identification
- **Pricing Analysts:** Elasticity modeling and optimization

---

## Contact

Built as a comprehensive competitive intelligence project showcasing:
- Data analysis & visualization
- Statistical modeling (price elasticity, correlation analysis)
- Business analytics & strategy
- Interactive dashboard development

---

---

**If you find this project useful, please star the repository!**

````
