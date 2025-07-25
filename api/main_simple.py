"""
TechGyant Insights API - Ultra Simple Version
Guaranteed to work on Render
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TechGyant Insights API",
    description="AI-powered investment recommendations for African tech startups",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class StartupInput(BaseModel):
    funding_stage: str
    sector: str
    country: str
    team_size: int
    years_since_founding: int
    monthly_revenue: float
    user_base: int

class InvestmentPrediction(BaseModel):
    predicted_investment: float
    confidence_score: float
    risk_level: str
    model_used: str

class StartupRecommendation(BaseModel):
    company_name: str
    country: str
    sector: str
    predicted_investment: float
    funding_stage: str
    confidence_score: float

# Sample data
SAMPLE_STARTUPS = [
    {
        "company_name": "KenyaTech Solutions",
        "country": "Kenya",
        "sector": "FinTech",
        "predicted_investment": 500000,
        "funding_stage": "Series A",
        "confidence_score": 0.85
    },
    {
        "company_name": "Lagos Health Innovation",
        "country": "Nigeria",
        "sector": "HealthTech",
        "predicted_investment": 750000,
        "funding_stage": "Series A",
        "confidence_score": 0.78
    },
    {
        "company_name": "Cape Town EdTech",
        "country": "South Africa",
        "sector": "EdTech",
        "predicted_investment": 300000,
        "funding_stage": "Seed",
        "confidence_score": 0.72
    },
    {
        "company_name": "Accra AgriTech",
        "country": "Ghana",
        "sector": "AgriTech",
        "predicted_investment": 400000,
        "funding_stage": "Seed",
        "confidence_score": 0.68
    },
    {
        "company_name": "Cairo CleanTech",
        "country": "Egypt",
        "sector": "CleanTech",
        "predicted_investment": 600000,
        "funding_stage": "Series A",
        "confidence_score": 0.80
    }
]

@app.get("/")
async def root():
    return {
        "message": "Welcome to TechGyant Insights API",
        "description": "AI-powered investment recommendations for African tech startups",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active",
        "endpoints": [
            "/health",
            "/predict",
            "/recommendations/country/{country}",
            "/recommendations/sector/{sector}",
            "/recommendations/top-countries",
            "/recommendations/africa-overview"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "TechGyant Insights API is running successfully",
        "sample_data_count": len(SAMPLE_STARTUPS)
    }

@app.post("/predict", response_model=InvestmentPrediction)
async def predict_investment(startup: StartupInput):
    """Predict investment amount for a startup"""
    
    try:
        # Simple prediction logic
        base_investment = 100000
        
        # Factor in different attributes
        stage_multipliers = {
            'Pre-Seed': 0.5,
            'Seed': 1,
            'Series A': 3,
            'Series B': 6,
            'Series C': 10,
            'Series D+': 15
        }
        
        sector_multipliers = {
            'FinTech': 1.5,
            'HealthTech': 1.3,
            'EdTech': 1.2,
            'AgriTech': 1.1,
            'CleanTech': 1.4,
            'E-commerce': 1.2,
            'Other': 1.0
        }
        
        stage_mult = stage_multipliers.get(startup.funding_stage, 1)
        sector_mult = sector_multipliers.get(startup.sector, 1.0)
        
        predicted_amount = (
            base_investment * stage_mult * sector_mult *
            (1 + startup.team_size / 50) *
            (1 + startup.years_since_founding / 10) *
            (1 + startup.monthly_revenue / 100000)
        )
        
        # Calculate confidence based on data completeness
        confidence = min(0.95, 0.6 + (startup.team_size / 100) + (startup.monthly_revenue / 500000))
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = "Low"
        elif confidence > 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return InvestmentPrediction(
            predicted_investment=round(predicted_amount, 2),
            confidence_score=round(confidence, 2),
            risk_level=risk_level,
            model_used="rule_based_v1"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/recommendations/country/{country}")
async def get_recommendations_by_country(country: str, limit: int = 10):
    """Get startup recommendations by country"""
    
    try:
        # Filter startups by country
        country_startups = [
            startup for startup in SAMPLE_STARTUPS
            if startup['country'].lower() == country.lower()
        ]
        
        if not country_startups:
            return {
                "recommendations": [],
                "total_count": 0,
                "average_investment": 0,
                "message": f"No startups found for {country}"
            }
        
        # Limit results
        limited_startups = country_startups[:limit]
        
        # Calculate average investment
        avg_investment = sum(s['predicted_investment'] for s in country_startups) / len(country_startups)
        
        return {
            "recommendations": limited_startups,
            "total_count": len(country_startups),
            "average_investment": round(avg_investment, 2),
            "country": country
        }
        
    except Exception as e:
        logger.error(f"Country recommendations error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/recommendations/sector/{sector}")
async def get_recommendations_by_sector(sector: str, limit: int = 10):
    """Get startup recommendations by sector"""
    
    try:
        # Filter startups by sector
        sector_startups = [
            startup for startup in SAMPLE_STARTUPS
            if startup['sector'].lower() == sector.lower()
        ]
        
        if not sector_startups:
            return {
                "recommendations": [],
                "total_count": 0,
                "average_investment": 0,
                "message": f"No startups found for {sector} sector"
            }
        
        # Limit results
        limited_startups = sector_startups[:limit]
        
        # Calculate average investment
        avg_investment = sum(s['predicted_investment'] for s in sector_startups) / len(sector_startups)
        
        return {
            "recommendations": limited_startups,
            "total_count": len(sector_startups),
            "average_investment": round(avg_investment, 2),
            "sector": sector
        }
        
    except Exception as e:
        logger.error(f"Sector recommendations error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/recommendations/top-countries")
async def get_top_countries(limit: int = 10):
    """Get top countries by investment activity"""
    
    try:
        # Group by country and calculate stats
        country_stats = {}
        
        for startup in SAMPLE_STARTUPS:
            country = startup['country']
            investment = startup['predicted_investment']
            
            if country not in country_stats:
                country_stats[country] = {
                    'country': country,
                    'total_investment': 0,
                    'startup_count': 0,
                    'avg_investment': 0
                }
            
            country_stats[country]['total_investment'] += investment
            country_stats[country]['startup_count'] += 1
        
        # Calculate averages and sort
        for country_data in country_stats.values():
            country_data['avg_investment'] = round(
                country_data['total_investment'] / country_data['startup_count'], 2
            )
        
        # Sort by total investment
        sorted_countries = sorted(
            country_stats.values(),
            key=lambda x: x['total_investment'],
            reverse=True
        )
        
        return {
            "top_countries": sorted_countries[:limit],
            "total_countries": len(country_stats)
        }
        
    except Exception as e:
        logger.error(f"Top countries error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get top countries: {str(e)}")

@app.get("/recommendations/africa-overview")
async def get_africa_overview():
    """Get Africa-wide investment overview"""
    
    try:
        total_startups = len(SAMPLE_STARTUPS)
        total_investment = sum(s['predicted_investment'] for s in SAMPLE_STARTUPS)
        avg_investment = total_investment / total_startups if total_startups > 0 else 0
        
        # Get sector distribution
        sector_counts = {}
        for startup in SAMPLE_STARTUPS:
            sector = startup['sector']
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Get country count
        countries = set(s['country'] for s in SAMPLE_STARTUPS)
        
        return {
            "total_startups": total_startups,
            "total_investment": round(total_investment, 2),
            "average_investment": round(avg_investment, 2),
            "top_sectors": dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)),
            "countries_covered": len(countries),
            "countries_list": sorted(list(countries))
        }
        
    except Exception as e:
        logger.error(f"Africa overview error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Africa overview: {str(e)}")

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
