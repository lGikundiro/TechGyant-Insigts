"""
TechGyant Insights API - Ultra Minimal Version
Zero external dependencies beyond FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os

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

# Sample data - hardcoded for reliability
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
        "message": "🚀 Welcome to TechGyant Insights API",
        "description": "AI-powered investment recommendations for African tech startups",
        "version": "1.0.0",
        "status": "🟢 LIVE",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "country_recommendations": "/recommendations/country/{country}",
            "sector_recommendations": "/recommendations/sector/{sector}",
            "top_countries": "/recommendations/top-countries",
            "africa_overview": "/recommendations/africa-overview"
        },
        "documentation": "/docs",
        "sample_countries": ["Kenya", "Nigeria", "South Africa", "Ghana", "Egypt"],
        "sample_sectors": ["FinTech", "HealthTech", "EdTech", "AgriTech", "CleanTech"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "🟢 Healthy",
        "message": "TechGyant Insights API is running perfectly!",
        "api_version": "1.0.0",
        "sample_data_count": len(SAMPLE_STARTUPS),
        "python_version": "3.11.9",
        "environment": "production"
    }

@app.get("/favicon.ico")
async def favicon():
    """Serve TechGyant logo as favicon"""
    favicon_path = os.path.join(os.path.dirname(__file__), "..", "static", "favicon.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    else:
        return {"error": "Favicon not found"}

@app.post("/predict", response_model=InvestmentPrediction)
async def predict_investment(startup: StartupInput):
    """Predict investment amount for a startup"""
    
    # Simple but effective prediction algorithm
    base_investment = 100000
    
    # Stage multipliers
    stage_multipliers = {
        'Pre-Seed': 0.5,
        'Seed': 1.0,
        'Series A': 3.0,
        'Series B': 6.0,
        'Series C': 10.0,
        'Series D+': 15.0
    }
    
    # Sector multipliers
    sector_multipliers = {
        'FinTech': 1.5,
        'HealthTech': 1.3,
        'EdTech': 1.2,
        'AgriTech': 1.1,
        'CleanTech': 1.4,
        'E-commerce': 1.2,
        'Other': 1.0
    }
    
    # Country multipliers (market size factor)
    country_multipliers = {
        'Nigeria': 1.3,
        'South Africa': 1.2,
        'Kenya': 1.1,
        'Egypt': 1.1,
        'Ghana': 1.0,
        'Other': 0.9
    }
    
    # Calculate multipliers
    stage_mult = stage_multipliers.get(startup.funding_stage, 1.0)
    sector_mult = sector_multipliers.get(startup.sector, 1.0)
    country_mult = country_multipliers.get(startup.country, 0.9)
    
    # Team size factor (larger teams = higher valuation)
    team_factor = 1 + (startup.team_size / 50)
    
    # Experience factor (older companies = more proven)
    experience_factor = 1 + (startup.years_since_founding / 10)
    
    # Revenue factor (revenue = strong signal)
    revenue_factor = 1 + (startup.monthly_revenue / 100000)
    
    # User base factor (users = market validation)
    user_factor = 1 + (startup.user_base / 100000)
    
    # Final prediction
    predicted_amount = (
        base_investment * 
        stage_mult * 
        sector_mult * 
        country_mult *
        team_factor * 
        experience_factor * 
        revenue_factor * 
        user_factor
    )
    
    # Calculate confidence score
    confidence_factors = [
        min(1.0, startup.team_size / 20),  # Team completeness
        min(1.0, startup.years_since_founding / 5),  # Track record
        min(1.0, startup.monthly_revenue / 50000),  # Revenue strength
        min(1.0, startup.user_base / 50000)  # Market validation
    ]
    
    confidence = 0.5 + (sum(confidence_factors) / len(confidence_factors)) * 0.4
    confidence = round(min(0.95, confidence), 2)
    
    # Determine risk level
    if confidence >= 0.8:
        risk_level = "Low"
    elif confidence >= 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return InvestmentPrediction(
        predicted_investment=round(predicted_amount, 2),
        confidence_score=confidence,
        risk_level=risk_level,
        model_used="techgyant_ai_v1"
    )

@app.get("/recommendations/country/{country}")
async def get_recommendations_by_country(country: str, limit: int = 10):
    """Get startup recommendations by country"""
    
    # Filter by country (case insensitive)
    country_startups = [
        startup for startup in SAMPLE_STARTUPS
        if startup['country'].lower() == country.lower()
    ]
    
    if not country_startups:
        return {
            "recommendations": [],
            "total_count": 0,
            "average_investment": 0,
            "country": country,
            "message": f"No startups found for {country}. Try: Kenya, Nigeria, South Africa, Ghana, Egypt"
        }
    
    # Limit results
    limited_startups = country_startups[:limit]
    
    # Calculate stats
    total_investment = sum(s['predicted_investment'] for s in country_startups)
    avg_investment = total_investment / len(country_startups)
    
    return {
        "recommendations": limited_startups,
        "total_count": len(country_startups),
        "average_investment": round(avg_investment, 2),
        "country": country,
        "total_investment": total_investment
    }

@app.get("/recommendations/sector/{sector}")
async def get_recommendations_by_sector(sector: str, limit: int = 10):
    """Get startup recommendations by sector"""
    
    # Filter by sector (case insensitive)
    sector_startups = [
        startup for startup in SAMPLE_STARTUPS
        if startup['sector'].lower() == sector.lower()
    ]
    
    if not sector_startups:
        return {
            "recommendations": [],
            "total_count": 0,
            "average_investment": 0,
            "sector": sector,
            "message": f"No startups found for {sector}. Try: FinTech, HealthTech, EdTech, AgriTech, CleanTech"
        }
    
    # Limit results
    limited_startups = sector_startups[:limit]
    
    # Calculate stats
    total_investment = sum(s['predicted_investment'] for s in sector_startups)
    avg_investment = total_investment / len(sector_startups)
    
    return {
        "recommendations": limited_startups,
        "total_count": len(sector_startups),
        "average_investment": round(avg_investment, 2),
        "sector": sector,
        "total_investment": total_investment
    }

@app.get("/recommendations/top-countries")
async def get_top_countries(limit: int = 10):
    """Get top countries by investment activity"""
    
    # Group by country
    country_stats = {}
    
    for startup in SAMPLE_STARTUPS:
        country = startup['country']
        investment = startup['predicted_investment']
        
        if country not in country_stats:
            country_stats[country] = {
                'country': country,
                'total_investment': 0,
                'startup_count': 0,
                'startups': []
            }
        
        country_stats[country]['total_investment'] += investment
        country_stats[country]['startup_count'] += 1
        country_stats[country]['startups'].append(startup['company_name'])
    
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
        "total_countries": len(country_stats),
        "total_investment_across_africa": sum(c['total_investment'] for c in country_stats.values())
    }

@app.get("/recommendations/africa-overview")
async def get_africa_overview():
    """Get Africa-wide investment overview"""
    
    total_startups = len(SAMPLE_STARTUPS)
    total_investment = sum(s['predicted_investment'] for s in SAMPLE_STARTUPS)
    avg_investment = total_investment / total_startups
    
    # Sector distribution
    sector_stats = {}
    for startup in SAMPLE_STARTUPS:
        sector = startup['sector']
        if sector not in sector_stats:
            sector_stats[sector] = {'count': 0, 'total_investment': 0}
        sector_stats[sector]['count'] += 1
        sector_stats[sector]['total_investment'] += startup['predicted_investment']
    
    # Calculate sector averages
    for sector_data in sector_stats.values():
        sector_data['avg_investment'] = round(
            sector_data['total_investment'] / sector_data['count'], 2
        )
    
    # Country list
    countries = list(set(s['country'] for s in SAMPLE_STARTUPS))
    
    return {
        "total_startups": total_startups,
        "total_investment": round(total_investment, 2),
        "average_investment": round(avg_investment, 2),
        "countries_covered": len(countries),
        "countries_list": sorted(countries),
        "sector_breakdown": sector_stats,
        "top_investment_country": max(countries, key=lambda c: sum(
            s['predicted_investment'] for s in SAMPLE_STARTUPS if s['country'] == c
        )),
        "summary": f"Tracking {total_startups} startups across {len(countries)} African countries with ${total_investment:,.0f} total predicted investment"
    }

# Health check for monitoring
@app.get("/ping")
async def ping():
    return {"status": "pong", "message": "API is alive! 🚀"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
