import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, filepath):
        """Load the raw dataset"""
        self.df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df
    
    def explore_data(self, save_plots=True):
        """Comprehensive data exploration and visualization"""
        print("=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Create visualizations directory if it doesn't exist
        if save_plots:
            os.makedirs("visualizations", exist_ok=True)
        
        # 1. Distribution of target variable
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.df['investor_readiness_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Investor Readiness Score')
        plt.xlabel('Investor Readiness Score')
        plt.ylabel('Frequency')
        
        # 2. Country distribution
        plt.subplot(2, 3, 2)
        country_counts = self.df['country'].value_counts()
        plt.pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Startups by Country')
        
        # 3. Sector distribution
        plt.subplot(2, 3, 3)
        sector_counts = self.df['sector'].value_counts()
        plt.bar(range(len(sector_counts)), sector_counts.values, color='lightcoral')
        plt.title('Startups by Sector')
        plt.xticks(range(len(sector_counts)), sector_counts.index, rotation=45, ha='right')
        plt.ylabel('Count')
        
        # 4. Correlation heatmap for numeric features
        plt.subplot(2, 3, 4)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 5. Readiness score by country
        plt.subplot(2, 3, 5)
        sns.boxplot(data=self.df, x='country', y='investor_readiness_score')
        plt.title('Investor Readiness Score by Country')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Readiness Score')
        
        # 6. Feature importance visualization (correlation with target)
        plt.subplot(2, 3, 6)
        target_corr = self.df[numeric_cols].corr()['investor_readiness_score'].drop('investor_readiness_score')
        target_corr_sorted = target_corr.abs().sort_values(ascending=True)
        
        colors = ['red' if x < 0 else 'green' for x in target_corr_sorted]
        plt.barh(range(len(target_corr_sorted)), target_corr_sorted.values, color=colors, alpha=0.7)
        plt.yticks(range(len(target_corr_sorted)), target_corr_sorted.index)
        plt.title('Feature Correlation with Target Variable')
        plt.xlabel('Absolute Correlation')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('visualizations/data_exploration.png', dpi=300, bbox_inches='tight')
            print("Exploration plots saved to visualizations/data_exploration.png")
        plt.show()
        
        # Print detailed statistics
        print("\n=== DETAILED STATISTICS ===")
        print("\nTarget variable statistics:")
        print(self.df['investor_readiness_score'].describe())
        
        print("\nTop correlations with target variable:")
        print(target_corr.abs().sort_values(ascending=False).head(10))
        
        return self.df
    
    def feature_engineering(self):
        """Engineer features and prepare data for modeling"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # 1. Create derived features
        print("Creating derived features...")
        
        # Funding per month ratio
        df_processed['funding_per_month'] = df_processed['funding_raised'] / (df_processed['months_in_operation'] + 1)
        
        # Team efficiency (funding per team member)
        df_processed['funding_per_team_member'] = df_processed['funding_raised'] / df_processed['team_size']
        
        # Social media engagement rate
        df_processed['media_to_followers_ratio'] = df_processed['media_coverage_count'] / (df_processed['social_media_followers'] + 1)
        
        # Experience-education combined score
        df_processed['founder_combined_score'] = (
            df_processed['founder_education_score'] * 0.6 + 
            df_processed['founder_experience_years'] * 0.4
        )
        
        # Market penetration indicator
        df_processed['market_penetration'] = (
            df_processed['customer_testimonials'] * df_processed['user_satisfaction_score'] / 
            (df_processed['months_in_operation'] + 1)
        )
        
        # 2. Encode categorical variables
        print("Encoding categorical variables...")
        categorical_cols = ['country', 'sector', 'problem_addressed']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        # 3. Handle outliers (using IQR method for key numeric features)
        print("Handling outliers...")
        numeric_features = [
            'founder_education_score', 'founder_experience_years', 'founder_network_score',
            'article_mentions', 'keyword_relevance_score', 'customer_testimonials',
            'user_satisfaction_score', 'market_size_estimate', 'problem_country_alignment',
            'months_in_operation', 'team_size', 'funding_raised', 'social_media_followers',
            'funding_per_month', 'funding_per_team_member', 'founder_combined_score'
        ]
        
        outliers_removed = 0
        for feature in numeric_features:
            Q1 = df_processed[feature].quantile(0.25)
            Q3 = df_processed[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df_processed[feature] < lower_bound) | (df_processed[feature] > upper_bound)
            outliers_count = outliers_mask.sum()
            outliers_removed += outliers_count
            
            # Cap outliers instead of removing them
            df_processed[feature] = df_processed[feature].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Capped {outliers_removed} outlier values")
        
        # 4. Select features for modeling
        self.feature_columns = [
            # Original numeric features
            'founder_education_score', 'founder_experience_years', 'founder_network_score',
            'article_mentions', 'keyword_relevance_score', 'sentiment_score',
            'customer_testimonials', 'user_satisfaction_score', 'market_size_estimate',
            'problem_country_alignment', 'months_in_operation', 'team_size',
            'funding_raised', 'social_media_followers', 'media_coverage_count',
            
            # Encoded categorical features
            'country_encoded', 'sector_encoded', 'problem_addressed_encoded',
            
            # Derived features
            'funding_per_month', 'funding_per_team_member', 'media_to_followers_ratio',
            'founder_combined_score', 'market_penetration'
        ]
        
        print(f"Selected {len(self.feature_columns)} features for modeling")
        
        # 5. Check for any remaining missing values
        missing_values = df_processed[self.feature_columns + ['investor_readiness_score']].isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            # Fill with median for numeric columns
            for col in self.feature_columns:
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        self.df_processed = df_processed
        print("Feature engineering completed!")
        
        return df_processed
    
    def prepare_for_modeling(self):
        """Prepare features and target for modeling"""
        print("\n=== PREPARING DATA FOR MODELING ===")
        
        # Separate features and target
        X = self.df_processed[self.feature_columns].copy()
        y = self.df_processed['investor_readiness_score'].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Standardize features
        print("Standardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        print("Data preparation completed!")
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def save_processed_data(self, filepath):
        """Save processed dataset"""
        self.df_processed.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    
    def get_feature_info(self):
        """Get information about processed features"""
        feature_info = {
            'feature_columns': self.feature_columns,
            'label_encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()},
            'total_features': len(self.feature_columns)
        }
        return feature_info

if __name__ == "__main__":
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load real startup data only
    real_data_path = "data/raw/techgyant_real_startups.csv"
    if not os.path.exists(real_data_path):
        print(f"Error: Real data file not found at {real_data_path}")
        print("Please ensure the real startup data file exists.")
        exit(1)
    
    # Load and process real data
    df = preprocessor.load_data(real_data_path)
    
    # Explore data
    df_explored = preprocessor.explore_data()
    
    # Feature engineering
    df_processed = preprocessor.feature_engineering()
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = preprocessor.prepare_for_modeling()
    
    # Save processed data
    preprocessor.save_processed_data("data/processed/techgyant_startups_processed.csv")
    
    # Print feature information
    feature_info = preprocessor.get_feature_info()
    print(f"\nFeature engineering summary:")
    print(f"Total features: {feature_info['total_features']}")
    print(f"Categorical encodings: {list(feature_info['label_encoders'].keys())}")
