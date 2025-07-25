"""
TechGyant Insights - Main Model Training Script
Comprehensive training pipeline for investor readiness prediction
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Import our custom modules
from data_preprocessing import DataPreprocessor
from linear_regression import LinearRegressionModel
from model_comparison import ModelComparison

def main():
    """Main training pipeline"""
    print("🚀 TechGyant Insights - Investor Readiness Prediction Model")
    print("=" * 70)
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Step 1: Load real TechGyant dataset
    print("\n📊 STEP 1: Dataset Preparation")
    
    # Check if we have real TechGyant data
    if os.path.exists("data/raw/techgyant_real_startups.csv"):
        print("Loading PURE REAL TechGyant startup dataset...")
        df = pd.read_csv("data/raw/techgyant_real_startups.csv")
        print(f"Real TechGyant startups loaded: {len(df)} companies")
        
        # If we have very few real startups, we need more data for ML
        if len(df) < 10:
            print(f"⚠️  Only {len(df)} real startups found. Need at least 10 for ML training.")
            print("Please run: python src/techgyant_data_extractor.py to get more data")
            return None, None, None, None
        
    else:
        print("❌ No real TechGyant data found. Please run the data extractor first:")
        print("python src/techgyant_data_extractor.py")
        print("\nCannot proceed without real data. Exiting...")
        return None, None, None, None
    
    # Step 2: Data preprocessing and feature engineering
    print("\n🔧 STEP 2: Data Preprocessing & Feature Engineering")
    preprocessor = DataPreprocessor()
    
    # Load the dataset (prioritize real data)
    if os.path.exists("data/raw/techgyant_real_startups.csv"):
        df = preprocessor.load_data("data/raw/techgyant_real_startups.csv")
    else:
        raise FileNotFoundError("No dataset found! Please run: python src/techgyant_data_extractor.py")
    
    # Explore and visualize data
    df_explored = preprocessor.explore_data(save_plots=True)
    
    # Feature engineering
    df_processed = preprocessor.feature_engineering()
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = preprocessor.prepare_for_modeling()
    
    # Save processed data
    preprocessor.save_processed_data("data/processed/techgyant_startups_processed.csv")
    
    # Step 3: Train Linear Regression with Gradient Descent
    print("\n🤖 STEP 3: Linear Regression Training")
    lr_model = LinearRegressionModel(learning_rate=0.01, max_iterations=1000)
    
    # Train with gradient descent
    lr_model.gradient_descent_fit(X_train, y_train, X_test, y_test)
    
    # Train with scikit-learn for comparison
    lr_model.sklearn_fit(X_train, y_train)
    
    # Evaluate linear regression
    metrics, feature_importance, y_train_pred_gd, y_test_pred_gd, y_train_pred_sk, y_test_pred_sk = \
        lr_model.evaluate_model(X_train, y_train, X_test, y_test, preprocessor.feature_columns)
    
    # Create visualizations
    lr_model.plot_results(X_train, y_train, X_test, y_test, 
                         y_train_pred_gd, y_test_pred_gd, y_train_pred_sk, y_test_pred_sk,
                         feature_importance, save_plots=True)
    
    # Step 4: Model Comparison
    print("\n📈 STEP 4: Model Comparison")
    comparison = ModelComparison()
    
    # Train other models
    dt_model = comparison.train_decision_tree(X_train, y_train)
    rf_model = comparison.train_random_forest(X_train, y_train)
    
    # Evaluate all models
    all_results = comparison.evaluate_all_models(X_train, y_train, X_test, y_test, lr_model)
    
    # Create comparison plots
    comparison.plot_comparison(X_test, y_test, preprocessor.feature_columns, save_plots=True)
    
    # Get results summary
    results_df = comparison.create_results_summary()
    
    # Step 5: Save Best Model
    print("\n💾 STEP 5: Saving Best Model")
    best_model_name, best_model, best_score = comparison.find_best_model()
    
    model_path = f"data/models/best_model_{best_model_name}.pkl"
    comparison.save_best_model(best_model_name, best_model, model_path)
    
    # Save preprocessing components
    preprocessing_components = {
        'scaler': preprocessor.scaler,
        'label_encoders': preprocessor.label_encoders,
        'feature_columns': preprocessor.feature_columns,
        'best_model_name': best_model_name,
        'model_path': model_path
    }
    joblib.dump(preprocessing_components, "data/models/preprocessing_components.pkl")
    print("Preprocessing components saved to data/models/preprocessing_components.pkl")
    
    # Step 6: Create Prediction Function
    print("\n🎯 STEP 6: Creating Prediction Service")
    create_prediction_service(best_model_name, model_path, preprocessing_components)
    
    print("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Best model: {best_model_name} (Test R² = {best_score:.4f})")
    print(f"Model saved to: {model_path}")
    print("Visualizations saved to: visualizations/")
    
    return best_model_name, best_model, preprocessor, results_df

def create_prediction_service(best_model_name, model_path, preprocessing_components):
    """Create prediction service functions"""
    
    prediction_service_code = f'''
"""
TechGyant Insights - Prediction Service
Functions to make predictions using the best trained model
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class StartupPredictor:
    def __init__(self, model_path="data/models/best_model_{best_model_name}.pkl", 
                 preprocessing_path="data/models/preprocessing_components.pkl"):
        """Initialize the predictor with trained model and preprocessing components"""
        
        # Load preprocessing components
        self.preprocessing_components = joblib.load(preprocessing_path)
        self.scaler = self.preprocessing_components['scaler']
        self.label_encoders = self.preprocessing_components['label_encoders']
        self.feature_columns = self.preprocessing_components['feature_columns']
        self.best_model_name = self.preprocessing_components['best_model_name']
        
        # Load the trained model
        if self.best_model_name == 'linear_regression':
            self.model_data = joblib.load(model_path)
            self.weights = self.model_data['weights']
            self.bias = self.model_data['bias']
            self.sklearn_model = self.model_data.get('sklearn_model', None)
        else:
            self.model = joblib.load(model_path)
        
        print(f"Loaded {{self.best_model_name}} model for predictions")
    
    def preprocess_input(self, startup_data):
        """Preprocess input data for prediction"""
        
        # Convert single record to DataFrame if needed
        if isinstance(startup_data, dict):
            df = pd.DataFrame([startup_data])
        else:
            df = startup_data.copy()
        
        # Create derived features (same as in preprocessing)
        df['funding_per_month'] = df['funding_raised'] / (df['months_in_operation'] + 1)
        df['funding_per_team_member'] = df['funding_raised'] / df['team_size']
        df['media_to_followers_ratio'] = df['media_coverage_count'] / (df['social_media_followers'] + 1)
        df['founder_combined_score'] = (
            df['founder_education_score'] * 0.6 + 
            df['founder_experience_years'] * 0.4
        )
        df['market_penetration'] = (
            df['customer_testimonials'] * df['user_satisfaction_score'] / 
            (df['months_in_operation'] + 1)
        )
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle new categories that weren't in training
                df[f'{{col}}_encoded'] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                )
        
        # Select and order features
        X = df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, startup_data):
        """Make prediction for startup investor readiness"""
        
        # Preprocess input
        X_scaled = self.preprocess_input(startup_data)
        
        # Make prediction based on model type
        if self.best_model_name == 'linear_regression':
            # Use gradient descent weights
            prediction = np.dot(X_scaled, self.weights) + self.bias
        else:
            # Use sklearn model
            prediction = self.model.predict(X_scaled)
        
        # Ensure prediction is within valid range (0-100)
        prediction = np.clip(prediction, 0, 100)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def predict_with_confidence(self, startup_data):
        """Make prediction with confidence intervals (simplified)"""
        
        prediction = self.predict(startup_data)
        
        # Simple confidence estimation based on model performance
        # In a real scenario, you'd use more sophisticated methods
        confidence_interval = 5.0  # ±5 points based on typical model performance
        
        return {{
            'prediction': float(prediction),
            'confidence_lower': float(max(0, prediction - confidence_interval)),
            'confidence_upper': float(min(100, prediction + confidence_interval)),
            'confidence_interval': confidence_interval
        }}
    
    def predict_batch(self, startup_data_list):
        """Make predictions for multiple startups"""
        
        df = pd.DataFrame(startup_data_list)
        X_scaled = self.preprocess_input(df)
        
        if self.best_model_name == 'linear_regression':
            predictions = np.dot(X_scaled, self.weights) + self.bias
        else:
            predictions = self.model.predict(X_scaled)
        
        predictions = np.clip(predictions, 0, 100)
        return predictions.tolist()

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = StartupPredictor()
    
    # Example startup data
    sample_startup = {{
        'country': 'Rwanda',
        'sector': 'FinTech',
        'problem_addressed': 'Financial Inclusion',
        'founder_education_score': 8.5,
        'founder_experience_years': 5.2,
        'founder_network_score': 7.3,
        'article_mentions': 12,
        'keyword_relevance_score': 8.7,
        'sentiment_score': 0.75,
        'customer_testimonials': 8,
        'user_satisfaction_score': 8.9,
        'market_size_estimate': 50000000,
        'problem_country_alignment': 9.2,
        'months_in_operation': 18,
        'team_size': 12,
        'funding_raised': 250000,
        'social_media_followers': 5000,
        'media_coverage_count': 6
    }}
    
    # Make prediction
    try:
        score = predictor.predict(sample_startup)
        print(f"Predicted Investor Readiness Score: {{score:.2f}}/100")
        
        # Get prediction with confidence
        result = predictor.predict_with_confidence(sample_startup)
        print(f"Prediction: {{result['prediction']:.2f}}")
        print(f"Confidence Interval: [{{result['confidence_lower']:.2f}}, {{result['confidence_upper']:.2f}}]")
        
    except Exception as e:
        print(f"Error making prediction: {{e}}")
        print("Please ensure the model has been trained first by running model_training.py")
'''
    
    # Write prediction service to file
    with open("src/prediction_service.py", "w") as f:
        f.write(prediction_service_code)
    
    print("Prediction service created: src/prediction_service.py")

if __name__ == "__main__":
    main()
