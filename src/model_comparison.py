import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_decision_tree(self, X_train, y_train, **kwargs):
        """Train Decision Tree Regressor"""
        print("\n=== TRAINING DECISION TREE ===")
        
        # Default parameters
        params = {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        params.update(kwargs)
        
        dt_model = DecisionTreeRegressor(**params)
        dt_model.fit(X_train, y_train)
        
        self.models['decision_tree'] = dt_model
        print(f"Decision Tree trained with parameters: {params}")
        return dt_model
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest Regressor"""
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        rf_model = RandomForestRegressor(**params)
        rf_model.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_model
        print(f"Random Forest trained with parameters: {params}")
        return rf_model
    
    def evaluate_all_models(self, X_train, y_train, X_test, y_test, linear_model):
        """Evaluate all models and compare performance"""
        print("\n=== EVALUATING ALL MODELS ===")
        
        # Add linear regression model
        self.models['linear_regression'] = linear_model
        
        # Store predictions and metrics
        self.predictions = {}
        self.results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            if model_name == 'linear_regression':
                # Use gradient descent predictions
                y_train_pred = model.predict_gradient_descent(X_train)
                y_test_pred = model.predict_gradient_descent(X_test)
            else:
                # Use standard sklearn predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            
            # Store predictions
            self.predictions[model_name] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Store results
            self.results[model_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
            
            print(f"  Train RMSE: {np.sqrt(train_mse):.4f}")
            print(f"  Test RMSE: {np.sqrt(test_mse):.4f}")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
        
        return self.results
    
    def find_best_model(self):
        """Determine the best performing model based on test R²"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name]['test_r2']
        
        print(f"\n=== BEST MODEL ===")
        print(f"Best model: {best_model_name}")
        print(f"Test R² score: {best_score:.4f}")
        
        return best_model_name, best_model, best_score
    
    def plot_comparison(self, X_test, y_test, feature_names, save_plots=True):
        """Create comprehensive comparison plots"""
        print("\n=== CREATING COMPARISON PLOTS ===")
        
        if save_plots:
            os.makedirs("visualizations", exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Comparison: Linear Regression vs Decision Tree vs Random Forest', 
                    fontsize=16, fontweight='bold')
        
        colors = {'linear_regression': 'blue', 'decision_tree': 'green', 'random_forest': 'red'}
        
        # 1. Performance metrics comparison
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        metric_names = ['Test R²', 'Test RMSE', 'Test MAE']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[0, i]
            model_names = list(self.results.keys())
            values = [self.results[model][metric] for model in model_names]
            
            bars = ax.bar(model_names, values, 
                         color=[colors[model] for model in model_names], 
                         alpha=0.7)
            ax.set_title(f'{name} Comparison')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Actual vs Predicted scatter plots
        for i, model_name in enumerate(self.models.keys()):
            ax = axes[1, i]
            y_pred = self.predictions[model_name]['test']
            
            ax.scatter(y_test, y_pred, alpha=0.6, color=colors[model_name], s=30)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'k--', linewidth=2, alpha=0.8)
            
            ax.set_title(f'{model_name.replace("_", " ").title()}\nActual vs Predicted')
            ax.set_xlabel('Actual Score')
            ax.set_ylabel('Predicted Score')
            ax.grid(True, alpha=0.3)
            
            # Add R² score to plot
            r2 = self.results[model_name]['test_r2']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
            print("Comparison plots saved to visualizations/model_comparison.png")
        plt.show()
        
        # Feature importance comparison for tree-based models
        if save_plots:
            self.plot_feature_importance(feature_names)
    
    def plot_feature_importance(self, feature_names, save_plots=True):
        """Plot feature importance for tree-based models"""
        plt.figure(figsize=(15, 10))
        
        # Get feature importances
        importances = {}
        for model_name, model in self.models.items():
            if model_name in ['decision_tree', 'random_forest']:
                importances[model_name] = model.feature_importances_
            elif model_name == 'linear_regression':
                # Use absolute coefficients as "importance"
                importances[model_name] = np.abs(model.weights)
        
        # Create subplots for each model
        n_models = len(importances)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        colors = {'linear_regression': 'blue', 'decision_tree': 'green', 'random_forest': 'red'}
        
        for i, (model_name, importance) in enumerate(importances.items()):
            # Get top 15 features
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(15)
            
            axes[i].barh(range(len(feature_df)), feature_df['importance'], 
                        color=colors[model_name], alpha=0.7)
            axes[i].set_yticks(range(len(feature_df)))
            axes[i].set_yticklabels(feature_df['feature'], fontsize=10)
            axes[i].set_title(f'{model_name.replace("_", " ").title()}\nTop 15 Feature Importance')
            axes[i].set_xlabel('Importance')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('visualizations/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            print("Feature importance plots saved to visualizations/feature_importance_comparison.png")
        plt.show()
    
    def create_results_summary(self):
        """Create a summary table of all results"""
        print("\n=== RESULTS SUMMARY ===")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("Model Performance Summary:")
        print("=" * 60)
        print(results_df)
        
        # Rank models by test R²
        test_r2_ranking = results_df['test_r2'].sort_values(ascending=False)
        print(f"\nModel Ranking (by Test R²):")
        print("=" * 30)
        for i, (model, score) in enumerate(test_r2_ranking.items()):
            print(f"{i+1}. {model.replace('_', ' ').title()}: {score:.4f}")
        
        return results_df
    
    def save_best_model(self, best_model_name, best_model, save_path):
        """Save the best performing model"""
        os.makedirs("data/models", exist_ok=True)
        
        if best_model_name == 'linear_regression':
            # Save gradient descent model components
            model_data = {
                'model_type': 'linear_regression_gradient_descent',
                'weights': best_model.weights,
                'bias': best_model.bias,
                'learning_rate': best_model.learning_rate,
                'sklearn_model': best_model.sklearn_model
            }
            joblib.dump(model_data, save_path)
        else:
            # Save sklearn model
            joblib.dump(best_model, save_path)
        
        print(f"Best model ({best_model_name}) saved to {save_path}")
        return save_path

if __name__ == "__main__":
    # This will be run from model_training.py
    pass
