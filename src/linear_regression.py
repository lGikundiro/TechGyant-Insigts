import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.test_losses = []
        self.sklearn_model = LinearRegression()
        
    def gradient_descent_fit(self, X_train, y_train, X_test=None, y_test=None):
        """Implement linear regression using gradient descent"""
        print("\n=== TRAINING LINEAR REGRESSION WITH GRADIENT DESCENT ===")
        
        # Initialize parameters
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Convert to numpy arrays if they aren't already
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        if X_test is not None and y_test is not None:
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
        
        print(f"Training samples: {n_samples}, Features: {n_features}")
        print(f"Learning rate: {self.learning_rate}, Max iterations: {self.max_iterations}")
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred_train = np.dot(X_train_np, self.weights) + self.bias
            
            # Calculate cost (MSE)
            train_loss = np.mean((y_train_np - y_pred_train) ** 2)
            self.train_losses.append(train_loss)
            
            # Calculate test loss if test data provided
            if X_test is not None and y_test is not None:
                y_pred_test = np.dot(X_test_np, self.weights) + self.bias
                test_loss = np.mean((y_test_np - y_pred_test) ** 2)
                self.test_losses.append(test_loss)
            
            # Calculate gradients
            dw = (-2/n_samples) * np.dot(X_train_np.T, (y_train_np - y_pred_train))
            db = (-2/n_samples) * np.sum(y_train_np - y_pred_train)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if iteration > 0 and abs(self.train_losses[-2] - self.train_losses[-1]) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break
            
            # Print progress
            if iteration % 100 == 0:
                test_loss_str = f", Test Loss: {test_loss:.4f}" if X_test is not None else ""
                print(f"Iteration {iteration}: Train Loss: {train_loss:.4f}{test_loss_str}")
        
        print(f"Training completed in {len(self.train_losses)} iterations")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        if self.test_losses:
            print(f"Final test loss: {self.test_losses[-1]:.4f}")
    
    def sklearn_fit(self, X_train, y_train):
        """Fit using scikit-learn for comparison"""
        print("\n=== TRAINING WITH SCIKIT-LEARN ===")
        self.sklearn_model.fit(X_train, y_train)
        print("Scikit-learn model trained successfully")
    
    def predict_gradient_descent(self, X):
        """Make predictions using gradient descent model"""
        X_np = X.values if hasattr(X, 'values') else X
        return np.dot(X_np, self.weights) + self.bias
    
    def predict_sklearn(self, X):
        """Make predictions using scikit-learn model"""
        return self.sklearn_model.predict(X)
    
    def evaluate_model(self, X_train, y_train, X_test, y_test, feature_names):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        # Make predictions
        y_train_pred_gd = self.predict_gradient_descent(X_train)
        y_test_pred_gd = self.predict_gradient_descent(X_test)
        y_train_pred_sk = self.predict_sklearn(X_train)
        y_test_pred_sk = self.predict_sklearn(X_test)
        
        # Calculate metrics
        metrics = {
            'gradient_descent': {
                'train_mse': mean_squared_error(y_train, y_train_pred_gd),
                'test_mse': mean_squared_error(y_test, y_test_pred_gd),
                'train_r2': r2_score(y_train, y_train_pred_gd),
                'test_r2': r2_score(y_test, y_test_pred_gd),
                'train_mae': mean_absolute_error(y_train, y_train_pred_gd),
                'test_mae': mean_absolute_error(y_test, y_test_pred_gd)
            },
            'sklearn': {
                'train_mse': mean_squared_error(y_train, y_train_pred_sk),
                'test_mse': mean_squared_error(y_test, y_test_pred_sk),
                'train_r2': r2_score(y_train, y_train_pred_sk),
                'test_r2': r2_score(y_test, y_test_pred_sk),
                'train_mae': mean_absolute_error(y_train, y_train_pred_sk),
                'test_mae': mean_absolute_error(y_test, y_test_pred_sk)
            }
        }
        
        # Print results
        print("Gradient Descent Results:")
        print(f"  Train MSE: {metrics['gradient_descent']['train_mse']:.4f}")
        print(f"  Test MSE: {metrics['gradient_descent']['test_mse']:.4f}")
        print(f"  Train R²: {metrics['gradient_descent']['train_r2']:.4f}")
        print(f"  Test R²: {metrics['gradient_descent']['test_r2']:.4f}")
        print(f"  Train MAE: {metrics['gradient_descent']['train_mae']:.4f}")
        print(f"  Test MAE: {metrics['gradient_descent']['test_mae']:.4f}")
        
        print("\nScikit-learn Results:")
        print(f"  Train MSE: {metrics['sklearn']['train_mse']:.4f}")
        print(f"  Test MSE: {metrics['sklearn']['test_mse']:.4f}")
        print(f"  Train R²: {metrics['sklearn']['train_r2']:.4f}")
        print(f"  Test R²: {metrics['sklearn']['test_r2']:.4f}")
        print(f"  Train MAE: {metrics['sklearn']['train_mae']:.4f}")
        print(f"  Test MAE: {metrics['sklearn']['test_mae']:.4f}")
        
        # Feature importance (coefficients)
        print("\nFeature Importance (Top 10):")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'gradient_descent_coef': self.weights,
            'sklearn_coef': self.sklearn_model.coef_
        })
        feature_importance['abs_gd_coef'] = abs(feature_importance['gradient_descent_coef'])
        feature_importance_sorted = feature_importance.sort_values('abs_gd_coef', ascending=False)
        print(feature_importance_sorted.head(10))
        
        return metrics, feature_importance_sorted, y_train_pred_gd, y_test_pred_gd, y_train_pred_sk, y_test_pred_sk
    
    def plot_results(self, X_train, y_train, X_test, y_test, y_train_pred_gd, y_test_pred_gd, 
                    y_train_pred_sk, y_test_pred_sk, feature_importance, save_plots=True):
        """Create comprehensive visualization plots"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        if save_plots:
            os.makedirs("visualizations", exist_ok=True)
        
        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('TechGyant Insights - Linear Regression Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Loss curves
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)
        if self.test_losses:
            axes[0, 0].plot(self.test_losses, label='Test Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Training & Test Loss Curves')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Mean Squared Error')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Gradient Descent - Train)
        axes[0, 1].scatter(y_train, y_train_pred_gd, alpha=0.6, color='blue', s=30)
        axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
        axes[0, 1].set_title('Actual vs Predicted (GD - Train)')
        axes[0, 1].set_xlabel('Actual Investor Readiness Score')
        axes[0, 1].set_ylabel('Predicted Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted (Gradient Descent - Test)
        axes[0, 2].scatter(y_test, y_test_pred_gd, alpha=0.6, color='green', s=30)
        axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        axes[0, 2].set_title('Actual vs Predicted (GD - Test)')
        axes[0, 2].set_xlabel('Actual Investor Readiness Score')
        axes[0, 2].set_ylabel('Predicted Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residuals plot (Gradient Descent)
        residuals_train = y_train - y_train_pred_gd
        residuals_test = y_test - y_test_pred_gd
        axes[1, 0].scatter(y_train_pred_gd, residuals_train, alpha=0.6, color='blue', s=30, label='Train')
        axes[1, 0].scatter(y_test_pred_gd, residuals_test, alpha=0.6, color='green', s=30, label='Test')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Residuals Plot (Gradient Descent)')
        axes[1, 0].set_xlabel('Predicted Score')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature importance
        top_features = feature_importance.head(12)
        axes[1, 1].barh(range(len(top_features)), top_features['abs_gd_coef'], color='skyblue', alpha=0.8)
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1, 1].set_title('Top 12 Feature Importance (|Coefficients|)')
        axes[1, 1].set_xlabel('Absolute Coefficient Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Comparison: Gradient Descent vs Scikit-learn (Test Set)
        axes[1, 2].scatter(y_test_pred_sk, y_test_pred_gd, alpha=0.6, color='purple', s=30)
        axes[1, 2].plot([y_test_pred_sk.min(), y_test_pred_sk.max()], 
                       [y_test_pred_sk.min(), y_test_pred_sk.max()], 'r--', linewidth=2)
        axes[1, 2].set_title('Gradient Descent vs Scikit-learn Predictions')
        axes[1, 2].set_xlabel('Scikit-learn Predictions')
        axes[1, 2].set_ylabel('Gradient Descent Predictions')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Distribution of predictions
        axes[2, 0].hist(y_train, bins=30, alpha=0.5, label='Actual (Train)', color='blue')
        axes[2, 0].hist(y_train_pred_gd, bins=30, alpha=0.5, label='Predicted (Train)', color='red')
        axes[2, 0].set_title('Distribution: Actual vs Predicted (Train)')
        axes[2, 0].set_xlabel('Investor Readiness Score')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Prediction errors distribution
        errors_train = np.abs(y_train - y_train_pred_gd)
        errors_test = np.abs(y_test - y_test_pred_gd)
        axes[2, 1].hist(errors_train, bins=25, alpha=0.6, label='Train Errors', color='blue')
        axes[2, 1].hist(errors_test, bins=25, alpha=0.6, label='Test Errors', color='green')
        axes[2, 1].set_title('Absolute Prediction Errors Distribution')
        axes[2, 1].set_xlabel('Absolute Error')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Model performance metrics comparison
        metrics_data = {
            'Train R²': [r2_score(y_train, y_train_pred_gd), r2_score(y_train, y_train_pred_sk)],
            'Test R²': [r2_score(y_test, y_test_pred_gd), r2_score(y_test, y_test_pred_sk)],
            'Train RMSE': [np.sqrt(mean_squared_error(y_train, y_train_pred_gd)), 
                          np.sqrt(mean_squared_error(y_train, y_train_pred_sk))],
            'Test RMSE': [np.sqrt(mean_squared_error(y_test, y_test_pred_gd)), 
                         np.sqrt(mean_squared_error(y_test, y_test_pred_sk))]
        }
        
        x_pos = np.arange(len(metrics_data))
        width = 0.35
        
        gd_values = [metrics_data[key][0] for key in metrics_data.keys()]
        sk_values = [metrics_data[key][1] for key in metrics_data.keys()]
        
        axes[2, 2].bar(x_pos - width/2, gd_values, width, label='Gradient Descent', alpha=0.8, color='blue')
        axes[2, 2].bar(x_pos + width/2, sk_values, width, label='Scikit-learn', alpha=0.8, color='orange')
        axes[2, 2].set_title('Model Performance Comparison')
        axes[2, 2].set_xlabel('Metrics')
        axes[2, 2].set_ylabel('Value')
        axes[2, 2].set_xticks(x_pos)
        axes[2, 2].set_xticklabels(list(metrics_data.keys()), rotation=45, ha='right')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('visualizations/linear_regression_analysis.png', dpi=300, bbox_inches='tight')
            print("Analysis plots saved to visualizations/linear_regression_analysis.png")
        plt.show()
        
        # Create a simple before/after scatter plot
        plt.figure(figsize=(15, 6))
        
        # Before (without regression line)
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(y_test)), y_test, alpha=0.6, color='blue', s=40, label='Actual')
        plt.title('Before: Actual Investor Readiness Scores (Test Set)')
        plt.xlabel('Sample Index')
        plt.ylabel('Investor Readiness Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # After (with regression predictions)
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(y_test)), y_test, alpha=0.6, color='blue', s=40, label='Actual')
        plt.scatter(range(len(y_test)), y_test_pred_gd, alpha=0.6, color='red', s=40, label='Predicted (Linear Regression)')
        
        # Add regression line through the data
        sorted_indices = np.argsort(y_test_pred_gd)
        plt.plot(np.array(range(len(y_test)))[sorted_indices], 
                y_test_pred_gd[sorted_indices], 
                color='red', linewidth=2, alpha=0.8, label='Regression Line')
        
        plt.title('After: Linear Regression Predictions vs Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Investor Readiness Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('visualizations/before_after_regression.png', dpi=300, bbox_inches='tight')
            print("Before/after plots saved to visualizations/before_after_regression.png")
        plt.show()
    
    def save_model(self, filepath_gd, filepath_sk):
        """Save both models"""
        # Save gradient descent model
        gd_model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        joblib.dump(gd_model_data, filepath_gd)
        
        # Save scikit-learn model
        joblib.dump(self.sklearn_model, filepath_sk)
        
        print(f"Gradient descent model saved to {filepath_gd}")
        print(f"Scikit-learn model saved to {filepath_sk}")

if __name__ == "__main__":
    # This will be run from model_training.py
    pass
