import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_synthetic_startup_data(n_samples=1000):
    """
    Create synthetic African tech startup data with realistic features
    based on TechGyant article analysis patterns
    """
    np.random.seed(42)
    
    # Countries with tech ecosystems
    countries = ['Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Rwanda', 
                'Uganda', 'Tanzania', 'Senegal', 'Cameroon', 'Zimbabwe']
    
    # Sectors
    sectors = ['FinTech', 'AgriTech', 'HealthTech', 'EdTech', 'E-commerce', 
              'LogisTech', 'CleanTech', 'AI/ML', 'Blockchain', 'InsurTech']
    
    # Problem areas
    problems = ['Financial Inclusion', 'Healthcare Access', 'Education Gap', 
               'Agricultural Efficiency', 'Rural Connectivity', 'Women Empowerment',
               'Youth Employment', 'Digital Payment', 'Supply Chain', 'Energy Access']
    
    data = []
    
    for i in range(n_samples):
        # Basic info
        startup_id = f"STARTUP_{i:04d}"
        country = np.random.choice(countries)
        sector = np.random.choice(sectors)
        problem_addressed = np.random.choice(problems)
        
        # Founder features (0-10 scale)
        founder_education_score = np.random.normal(6, 2)  # Education background
        founder_experience_years = np.random.exponential(3)  # Years of experience
        founder_network_score = np.random.normal(5, 2)  # Network strength
        
        # Article-derived features
        article_mentions = np.random.poisson(5)  # Number of article mentions
        keyword_relevance_score = np.random.normal(7, 2)  # Topic relevance (0-10)
        sentiment_score = np.random.normal(0.6, 0.3)  # Sentiment (-1 to 1)
        
        # Customer/Market features
        customer_testimonials = np.random.poisson(3)  # Number of testimonials
        user_satisfaction_score = np.random.normal(7, 2)  # User satisfaction (0-10)
        market_size_estimate = np.random.lognormal(15, 1)  # Market size (log scale)
        
        # Problem-country alignment (0-10)
        # Some combinations are more aligned
        alignment_base = 5
        if (country == 'Rwanda' and problem_addressed == 'Financial Inclusion') or \
           (country == 'Nigeria' and problem_addressed == 'Digital Payment') or \
           (country == 'Kenya' and problem_addressed == 'Agricultural Efficiency'):
            alignment_base = 8
        elif (country == 'South Africa' and problem_addressed == 'Energy Access') or \
             (country == 'Ghana' and problem_addressed == 'Healthcare Access'):
            alignment_base = 7
        
        problem_country_alignment = np.random.normal(alignment_base, 1.5)
        
        # Traction indicators
        months_in_operation = np.random.exponential(18)
        team_size = np.random.poisson(8) + 1
        funding_raised = np.random.lognormal(10, 2)  # In USD
        
        # Social media presence
        social_media_followers = np.random.lognormal(8, 2)
        media_coverage_count = np.random.poisson(2)
        
        # Ensure values are within reasonable ranges
        founder_education_score = np.clip(founder_education_score, 0, 10)
        founder_experience_years = np.clip(founder_experience_years, 0, 20)
        founder_network_score = np.clip(founder_network_score, 0, 10)
        keyword_relevance_score = np.clip(keyword_relevance_score, 0, 10)
        sentiment_score = np.clip(sentiment_score, -1, 1)
        user_satisfaction_score = np.clip(user_satisfaction_score, 0, 10)
        problem_country_alignment = np.clip(problem_country_alignment, 0, 10)
        months_in_operation = np.clip(months_in_operation, 1, 120)
        
        # Calculate target variable: Investor Readiness Score (0-100)
        # This is our composite score based on multiple factors
        readiness_score = (
            founder_education_score * 0.15 +
            founder_experience_years * 0.12 +
            founder_network_score * 0.10 +
            keyword_relevance_score * 0.08 +
            (sentiment_score + 1) * 5 * 0.08 +  # Convert to 0-10 scale
            customer_testimonials * 0.12 +
            user_satisfaction_score * 0.15 +
            np.log(market_size_estimate) * 0.05 +
            problem_country_alignment * 0.15 +
            np.random.normal(0, 5)  # Add some noise
        )
        
        # Scale to 0-100 and add realistic constraints
        readiness_score = np.clip(readiness_score * 8 + 20, 0, 100)
        
        data.append({
            'startup_id': startup_id,
            'country': country,
            'sector': sector,
            'problem_addressed': problem_addressed,
            'founder_education_score': round(founder_education_score, 2),
            'founder_experience_years': round(founder_experience_years, 1),
            'founder_network_score': round(founder_network_score, 2),
            'article_mentions': article_mentions,
            'keyword_relevance_score': round(keyword_relevance_score, 2),
            'sentiment_score': round(sentiment_score, 3),
            'customer_testimonials': customer_testimonials,
            'user_satisfaction_score': round(user_satisfaction_score, 2),
            'market_size_estimate': round(market_size_estimate, 0),
            'problem_country_alignment': round(problem_country_alignment, 2),
            'months_in_operation': round(months_in_operation, 0),
            'team_size': team_size,
            'funding_raised': round(funding_raised, 0),
            'social_media_followers': round(social_media_followers, 0),
            'media_coverage_count': media_coverage_count,
            'investor_readiness_score': round(readiness_score, 2)
        })
    
    return pd.DataFrame(data)

def save_dataset(df, filepath):
    """Save dataset to CSV"""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
    
if __name__ == "__main__":
    # Create the dataset
    print("Creating TechGyant FundScout startup dataset...")
    df = create_synthetic_startup_data(1000)
    
    # Save raw data
    save_dataset(df, "data/raw/techgyant_startups.csv")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
