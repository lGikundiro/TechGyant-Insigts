"""
TechGyant Insights API
FastAPI application for African tech startup investor readiness predictions
"""

from typing import List
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Import our models
from models import (
    StartupPredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    StartupRecommendation,
    CountryRecommendationsResponse,
    SectorRecommendationsResponse,
    CountrySectorRecommendationsResponse,
    TopCountriesResponse,
    AfricaOverviewResponse
)

# Add parent directory to path to import our prediction service
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from prediction_service import StartupPredictor
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import prediction service: {e}")
    print("Please run model training first: python src/model_training.py")
    PREDICTOR_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="TechGyant Insights API",
    description="AI-powered African tech startup investor readiness prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None
startup_data = None  # Cache for startup data

def load_startup_data():
    """Load startup data for recommendations"""
    global startup_data
    
    if startup_data is None:
        try:
            # Load only real startup data
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "techgyant_real_startups.csv")
            if os.path.exists(data_path):
                startup_data = pd.read_csv(data_path)
            else:
                startup_data = pd.DataFrame()  # Empty dataframe if no data
            
            print(f"Loaded {len(startup_data)} startups for recommendations")
        except Exception as e:
            print(f"Error loading startup data: {e}")
            startup_data = pd.DataFrame()
    
    return startup_data

def format_startup_recommendation(row):
    """Convert dataframe row to StartupRecommendation model"""
    return StartupRecommendation(
        startup_id=row.get('startup_id', 'N/A'),
        startup_name=row.get('startup_name', row.get('startup_id', 'Unknown')),
        country=row.get('country', 'Unknown'),
        sector=row.get('sector', 'Unknown'), 
        problem_addressed=row.get('problem_addressed', 'Unknown'),
        investor_readiness_score=round(float(row.get('investor_readiness_score', 0)), 2),
        funding_raised=float(row.get('funding_raised', 0)),
        team_size=int(row.get('team_size', 0)),
        months_in_operation=int(row.get('months_in_operation', 0)),
        problem_country_alignment=round(float(row.get('problem_country_alignment', 0)), 2),
        user_satisfaction_score=round(float(row.get('user_satisfaction_score', 0)), 2),
        valuation_usd=float(row.get('valuation_usd', 0)) if pd.notna(row.get('valuation_usd')) else None,
        recent_milestone=row.get('recent_milestone') if pd.notna(row.get('recent_milestone')) else None
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the ML model and data on startup"""
    global predictor, PREDICTOR_AVAILABLE, startup_data
    
    print("🚀 Starting TechGyant Insights API...")
    
    # Load startup data first
    try:
        startup_data = load_startup_data()
        if not startup_data.empty:
            print(f"✅ Loaded {len(startup_data)} startups for recommendations")
        else:
            print("⚠️  No startup data loaded - using demo mode")
    except Exception as e:
        print(f"⚠️  Error loading startup data: {e}")
        startup_data = pd.DataFrame()
    
    # Try to load ML model (optional - API works without it)
    if PREDICTOR_AVAILABLE:
        try:
            model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models")
            if os.path.exists(model_path):
                predictor = StartupPredictor()
                print("✅ ML model loaded successfully")
            else:
                print("⚠️  Model files not found. Prediction endpoints disabled.")
                PREDICTOR_AVAILABLE = False
        except Exception as e:
            print(f"⚠️  Failed to load ML model: {e}")
            PREDICTOR_AVAILABLE = False
    
    print("🌐 API startup complete!")

def get_risk_level_and_recommendation(score: float) -> tuple[str, str]:
    """Determine risk level and recommendation based on score"""
    
    if score >= 80:
        risk_level = "Low"
        recommendation = "Excellent investment opportunity! Strong fundamentals across all metrics."
    elif score >= 70:
        risk_level = "Low-Medium"
        recommendation = "Very promising startup with strong potential. Recommended for investment."
    elif score >= 60:
        risk_level = "Medium"
        recommendation = "Good startup with solid foundations. Consider for investment with due diligence."
    elif score >= 50:
        risk_level = "Medium-High"
        recommendation = "Startup shows potential but requires careful evaluation of risks."
    elif score >= 40:
        risk_level = "High"
        recommendation = "Early-stage startup with challenges. High risk, potentially high reward."
    else:
        risk_level = "Very High"
        recommendation = "Significant challenges identified. Proceed with extreme caution."
    
    return risk_level, recommendation

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TechGyant FundScout API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .status.available {{ background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .status.unavailable {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            .endpoint {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
            .method {{ font-weight: bold; color: #007bff; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            ul {{ padding-left: 20px; }}
            li {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 TechGyant Insights API</h1>
            <p>AI-powered African tech startup investor readiness prediction API</p>
            
            <div class="status {'available' if PREDICTOR_AVAILABLE else 'unavailable'}">
                <strong>Model Status:</strong> {'✅ Available' if PREDICTOR_AVAILABLE else '❌ Not Available - Please train the model first'}
            </div>
            
            <h2>📋 Available Endpoints</h2>
            
            <div class="endpoint">
                <div class="method">GET /health</div>
                <p>Check API health and model status</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /predict</div>
                <p>Predict investor readiness score for a single startup</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /predict/batch</div>
                <p>Predict investor readiness scores for multiple startups (max 10)</p>
            </div>
            
            <h2>🌍 Investment Recommendation Endpoints</h2>
            
            <div class="endpoint">
                <div class="method">GET /recommendations/country/{country}</div>
                <p>Get recommended startups by country (e.g., /recommendations/country/Nigeria)</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /recommendations/sector/{sector}</div>
                <p>Get recommended startups by sector (e.g., /recommendations/sector/FinTech)</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /recommendations/country/{country}/sector/{sector}</div>
                <p>Get recommended startups by country AND sector</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /recommendations/top-countries</div>
                <p>Get top African countries for investment ranked by startup quality</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /recommendations/africa-overview</div>
                <p>Comprehensive overview of investment opportunities across Africa</p>
            </div>
            
            <h2>📖 Documentation</h2>
            <ul>
                <li><a href="/docs">Swagger UI Documentation</a></li>
                <li><a href="/redoc">ReDoc Documentation</a></li>
                <li><a href="/openapi.json">OpenAPI Schema</a></li>
            </ul>
            
            <h2>🌍 About TechGyant Insights</h2>
            <p>This API uses machine learning to analyze African tech startups and predict their investor readiness scores based on:</p>
            <ul>
                <li>Founder backgrounds and experience</li>
                <li>Market alignment and problem-solution fit</li>
                <li>Customer testimonials and satisfaction</li>
                <li>Media coverage and sentiment analysis</li>
                <li>Operational metrics and traction</li>
            </ul>
            
            <p><strong>Built with:</strong> FastAPI, Scikit-learn, Pydantic | <strong>Version:</strong> 1.0.0</p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if PREDICTOR_AVAILABLE else "degraded",
        message="API is running" if PREDICTOR_AVAILABLE else "API running but ML model not available",
        model_loaded=PREDICTOR_AVAILABLE,
        timestamp=datetime.now().isoformat()
    )

@app.get("/favicon.ico")
async def favicon():
    """Serve TechGyant logo as favicon"""
    favicon_path = os.path.join(os.path.dirname(__file__), "..", "static", "favicon.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    else:
        return {"error": "Favicon not found"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_startup(request: StartupPredictionRequest):
    """
    Predict investor readiness score for a startup based on country and/or sector
    
    Request can contain:
    - Only country
    - Only sector  
    - Both country and sector
    
    Returns a score from 0-100 indicating how ready the startup is for investment.
    """
    
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="ML model not available. Please ensure the model has been trained and loaded."
        )
    
    # Validate that at least one field is provided
    if not request.country and not request.sector:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'country' or 'sector' must be provided"
        )
    
    try:
        # Convert request to dictionary with only provided fields
        startup_data = {}
        if request.country:
            startup_data['country'] = request.country.value
        if request.sector:
            startup_data['sector'] = request.sector.value
        
        # For missing fields, use default/average values for prediction
        # This would need to be implemented in your prediction service
        result = predictor.predict_with_confidence(startup_data)
        
        # Get risk level and recommendation
        risk_level, recommendation = get_risk_level_and_recommendation(result['prediction'])
        
        return PredictionResponse(
            investor_readiness_score=round(result['prediction'], 2),
            confidence_lower=round(result['confidence_lower'], 2),
            confidence_upper=round(result['confidence_upper'], 2),
            risk_level=risk_level,
            recommendation=recommendation
        )
        
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_startups_batch(request: BatchPredictionRequest):
    """
    Predict investor readiness scores for multiple startups
    
    Accepts up to 10 startups and returns predictions for all of them,
    along with summary statistics.
    """
    
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please ensure the model has been trained and loaded."
        )
    
    try:
        # Process each startup
        predictions = []
        scores = []
        
        for startup_request in request.startups:
            startup_data = startup_request.dict()
            result = predictor.predict_with_confidence(startup_data)
            
            risk_level, recommendation = get_risk_level_and_recommendation(result['prediction'])
            
            prediction = PredictionResponse(
                investor_readiness_score=round(result['prediction'], 2),
                confidence_lower=round(result['confidence_lower'], 2),
                confidence_upper=round(result['confidence_upper'], 2),
                risk_level=risk_level,
                recommendation=recommendation
            )
            
            predictions.append(prediction)
            scores.append(result['prediction'])
        
        # Calculate summary statistics
        summary = {
            "total_startups": len(scores),
            "average_score": round(np.mean(scores), 2),
            "highest_score": round(max(scores), 2),
            "lowest_score": round(min(scores), 2),
            "standard_deviation": round(np.std(scores), 2),
            "recommended_count": len([s for s in scores if s >= 60]),
            "high_potential_count": len([s for s in scores if s >= 80])
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    
    if not PREDICTOR_AVAILABLE or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not available"
        )
    
    try:
        model_info = {
            "model_type": predictor.best_model_name,
            "features_count": len(predictor.feature_columns),
            "categorical_features": list(predictor.label_encoders.keys()),
            "prediction_range": "0-100",
            "confidence_interval": "±5 points",
            "last_trained": "Unknown"  # You could add timestamp tracking
        }
        
        return model_info
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

# ========================================
# INVESTMENT RECOMMENDATION ENDPOINTS
# ========================================

@app.get("/recommendations/country/{country}", response_model=CountryRecommendationsResponse)
async def get_country_recommendations(country: str, min_score: float = 60.0):
    """
    Get recommended startups by country
    
    Returns startups in the specified country with readiness score >= min_score
    """
    try:
        data = load_startup_data()
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No startup data available")
        
        # Filter by country (case insensitive)
        country_data = data[data['country'].str.lower() == country.lower()]
        
        if country_data.empty:
            raise HTTPException(status_code=404, detail=f"No startups found for country: {country}")
        
        # Filter by minimum score
        recommended = country_data[country_data['investor_readiness_score'] >= min_score]
        
        # Sort by readiness score (descending)
        recommended = recommended.sort_values('investor_readiness_score', ascending=False)
        
        # Get top sectors
        top_sectors = country_data['sector'].value_counts().head(3).index.tolist()
        
        # Format startups
        startups = [format_startup_recommendation(row) for _, row in recommended.iterrows()]
        
        return CountryRecommendationsResponse(
            country=country.title(),
            total_startups=len(country_data),
            recommended_startups=len(recommended),
            average_readiness_score=round(country_data['investor_readiness_score'].mean(), 2),
            top_sectors=top_sectors,
            startups=startups
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting country recommendations: {str(e)}")

@app.get("/recommendations/sector/{sector}", response_model=SectorRecommendationsResponse)
async def get_sector_recommendations(sector: str, min_score: float = 60.0):
    """
    Get recommended startups by sector
    
    Returns startups in the specified sector with readiness score >= min_score
    """
    try:
        data = load_startup_data()
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No startup data available")
        
        # Filter by sector (case insensitive)
        sector_data = data[data['sector'].str.lower() == sector.lower()]
        
        if sector_data.empty:
            raise HTTPException(status_code=404, detail=f"No startups found for sector: {sector}")
        
        # Filter by minimum score
        recommended = sector_data[sector_data['investor_readiness_score'] >= min_score]
        
        # Sort by readiness score (descending)
        recommended = recommended.sort_values('investor_readiness_score', ascending=False)
        
        # Get top countries
        top_countries = sector_data['country'].value_counts().head(3).index.tolist()
        
        # Format startups
        startups = [format_startup_recommendation(row) for _, row in recommended.iterrows()]
        
        return SectorRecommendationsResponse(
            sector=sector.title(),
            total_startups=len(sector_data),
            recommended_startups=len(recommended),
            average_readiness_score=round(sector_data['investor_readiness_score'].mean(), 2),
            top_countries=top_countries,
            startups=startups
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sector recommendations: {str(e)}")

@app.get("/recommendations/country/{country}/sector/{sector}", response_model=CountrySectorRecommendationsResponse)
async def get_country_sector_recommendations(country: str, sector: str, min_score: float = 60.0):
    """
    Get recommended startups by country AND sector
    
    Returns startups in the specified country and sector with readiness score >= min_score
    """
    try:
        data = load_startup_data()
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No startup data available")
        
        # Filter by both country and sector (case insensitive)
        filtered_data = data[
            (data['country'].str.lower() == country.lower()) & 
            (data['sector'].str.lower() == sector.lower())
        ]
        
        if filtered_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No startups found for {sector} sector in {country}"
            )
        
        # Filter by minimum score
        recommended = filtered_data[filtered_data['investor_readiness_score'] >= min_score]
        
        # Sort by readiness score (descending)
        recommended = recommended.sort_values('investor_readiness_score', ascending=False)
        
        # Format startups
        startups = [format_startup_recommendation(row) for _, row in recommended.iterrows()]
        
        return CountrySectorRecommendationsResponse(
            country=country.title(),
            sector=sector.title(),
            total_startups=len(filtered_data),
            average_readiness_score=round(filtered_data['investor_readiness_score'].mean(), 2),
            startups=startups
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting country+sector recommendations: {str(e)}")

@app.get("/recommendations/top-countries", response_model=List[TopCountriesResponse])
async def get_top_countries(limit: int = 5, min_score: float = 60.0):
    """
    Get top countries for investment based on startup quality and quantity
    
    Returns countries ranked by success rate and average readiness score
    """
    try:
        data = load_startup_data()
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No startup data available")
        
        # Group by country and calculate metrics
        country_stats = []
        
        for country in data['country'].unique():
            country_data = data[data['country'] == country]
            recommended = country_data[country_data['investor_readiness_score'] >= min_score]
            
            success_rate = (len(recommended) / len(country_data)) * 100
            
            # Get top sectors for this country
            top_sectors = country_data['sector'].value_counts().head(3).index.tolist()
            
            country_stats.append(TopCountriesResponse(
                country=country,
                total_startups=len(country_data),
                recommended_startups=len(recommended),
                average_readiness_score=round(country_data['investor_readiness_score'].mean(), 2),
                total_funding_raised=float(country_data['funding_raised'].sum()),
                top_sectors=top_sectors,
                success_rate=round(success_rate, 2)
            ))
        
        # Sort by success rate and average score
        country_stats.sort(key=lambda x: (x.success_rate, x.average_readiness_score), reverse=True)
        
        return country_stats[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting top countries: {str(e)}")

@app.get("/recommendations/africa-overview", response_model=AfricaOverviewResponse)
async def get_africa_overview(min_score: float = 60.0, top_startups_limit: int = 20):
    """
    Get comprehensive overview of investment opportunities across Africa
    
    Returns Africa-wide statistics and top recommended startups
    """
    try:
        data = load_startup_data()
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No startup data available")
        
        # Overall statistics
        total_startups = len(data)
        recommended = data[data['investor_readiness_score'] >= min_score]
        recommended_count = len(recommended)
        
        # Get top countries (for the overview)
        top_countries_data = []
        for country in data['country'].unique():
            country_data = data[data['country'] == country]
            country_recommended = country_data[country_data['investor_readiness_score'] >= min_score]
            success_rate = (len(country_recommended) / len(country_data)) * 100
            
            top_sectors = country_data['sector'].value_counts().head(3).index.tolist()
            
            top_countries_data.append(TopCountriesResponse(
                country=country,
                total_startups=len(country_data),
                recommended_startups=len(country_recommended),
                average_readiness_score=round(country_data['investor_readiness_score'].mean(), 2),
                total_funding_raised=float(country_data['funding_raised'].sum()),
                top_sectors=top_sectors,
                success_rate=round(success_rate, 2)
            ))
        
        # Sort countries by success rate
        top_countries_data.sort(key=lambda x: (x.success_rate, x.average_readiness_score), reverse=True)
        
        # Get top recommended startups across Africa
        top_startups = recommended.nlargest(top_startups_limit, 'investor_readiness_score')
        startups = [format_startup_recommendation(row) for _, row in top_startups.iterrows()]
        
        return AfricaOverviewResponse(
            total_startups=total_startups,
            recommended_startups=recommended_count,
            countries_covered=data['country'].nunique(),
            sectors_covered=data['sector'].nunique(),
            total_funding=float(data['funding_raised'].sum()),
            average_readiness_score=round(data['investor_readiness_score'].mean(), 2),
            top_countries=top_countries_data[:5],
            startups=startups
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Africa overview: {str(e)}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(
        status_code=422,
        detail=f"Invalid input value: {str(exc)}"
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "The requested endpoint does not exist"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
