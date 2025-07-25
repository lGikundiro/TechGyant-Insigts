"""
TechGyant Insights - Setup and Run Script
Complete setup and demonstration of the project
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

def setup_python_environment():
    """Configure Python environment and install packages"""
    print("🔧 Setting up Python environment...")
    
    # Required packages for the project
    packages = [
        'pandas==2.1.3',
        'numpy==1.24.3', 
        'scikit-learn==1.3.2',
        'matplotlib==3.8.2',
        'seaborn==0.13.0',
        'joblib==1.3.2',
        'fastapi==0.104.1',
        'uvicorn[standard]==0.24.0',
        'pydantic==2.5.0',
        'python-multipart==0.0.6'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def run_training_pipeline():
    """Run the complete training pipeline"""
    print("\n🚀 Running TechGyant Insights Training Pipeline...")
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Run the training script
    try:
        subprocess.check_call([sys.executable, "src/model_training.py"])
        print("✅ Training pipeline completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training pipeline failed: {e}")
        return False

def run_api_server():
    """Start the FastAPI server"""
    print("\n🌐 Starting TechGyant Insights API server...")
    
    try:
        # Change to API directory
        api_dir = "api"
        os.chdir(api_dir)
        
        print("Starting server on http://localhost:8000")
        print("API Documentation: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        subprocess.check_call([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")

def test_api():
    """Test the API with sample requests"""
    print("\n🧪 Testing API endpoints...")
    
    try:
        import requests
        import json
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        print("Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health")
        print(f"Health status: {health_response.status_code}")
        
        # Test prediction endpoint
        print("Testing prediction endpoint...")
        sample_data = {
            "country": "Rwanda",
            "sector": "FinTech", 
            "problem_addressed": "Financial Inclusion",
            "founder_education_score": 8.5,
            "founder_experience_years": 5.2,
            "founder_network_score": 7.3,
            "article_mentions": 12,
            "keyword_relevance_score": 8.7,
            "sentiment_score": 0.75,
            "customer_testimonials": 8,
            "user_satisfaction_score": 8.9,
            "market_size_estimate": 50000000,
            "problem_country_alignment": 9.2,
            "months_in_operation": 18,
            "team_size": 12,
            "funding_raised": 250000,
            "social_media_followers": 5000,
            "media_coverage_count": 6
        }
        
        prediction_response = requests.post(
            f"{base_url}/predict", 
            json=sample_data
        )
        
        if prediction_response.status_code == 200:
            result = prediction_response.json()
            print(f"✅ Prediction successful!")
            print(f"Investor Readiness Score: {result['investor_readiness_score']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Recommendation: {result['recommendation']}")
        else:
            print(f"❌ Prediction failed: {prediction_response.status_code}")
            
    except ImportError:
        print("❌ Requests library not available for testing")
    except Exception as e:
        print(f"❌ API test failed: {e}")

def main():
    """Main setup and run function"""
    print("🚀 TechGyant Insights - Complete Setup and Demo")
    print("=" * 60)
    
    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Setup environment and install packages")
    print("2. Run training pipeline only") 
    print("3. Start API server")
    print("4. Complete setup (install + train + start server)")
    print("5. Test API (requires server to be running)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        setup_python_environment()
        
    elif choice == "2":
        if run_training_pipeline():
            print("\n✅ Training completed! You can now start the API server.")
        
    elif choice == "3":
        print("\n⚠️  Make sure you've trained the model first!")
        run_api_server()
        
    elif choice == "4":
        setup_python_environment()
        if run_training_pipeline():
            print("\n🎉 Setup complete! Starting API server...")
            run_api_server()
        
    elif choice == "5":
        test_api()
        
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
