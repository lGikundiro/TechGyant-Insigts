"""
TechGyant Insights API - Pydantic Models
Data validation models for the FastAPI application
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from enum import Enum

class CountryEnum(str, Enum):
    nigeria = "Nigeria"
    kenya = "Kenya"
    south_africa = "South Africa"
    ghana = "Ghana"
    rwanda = "Rwanda"
    uganda = "Uganda"
    tanzania = "Tanzania"
    senegal = "Senegal"
    cameroon = "Cameroon"
    zimbabwe = "Zimbabwe"

class SectorEnum(str, Enum):
    fintech = "FinTech"
    agritech = "AgriTech"
    healthtech = "HealthTech"
    edtech = "EdTech"
    ecommerce = "E-commerce"
    logistech = "LogisTech"
    cleantech = "CleanTech"
    aiml = "AI/ML"
    blockchain = "Blockchain"
    insurtech = "InsurTech"

class ProblemEnum(str, Enum):
    financial_inclusion = "Financial Inclusion"
    healthcare_access = "Healthcare Access"
    education_gap = "Education Gap"
    agricultural_efficiency = "Agricultural Efficiency"
    rural_connectivity = "Rural Connectivity"
    women_empowerment = "Women Empowerment"
    youth_employment = "Youth Employment"
    digital_payment = "Digital Payment"
    supply_chain = "Supply Chain"
    energy_access = "Energy Access"

class StartupPredictionRequest(BaseModel):
    """Model for startup prediction request"""
    
    # Basic Information
    country: CountryEnum = Field(..., description="Country where the startup operates")
    sector: SectorEnum = Field(..., description="Primary business sector")
    problem_addressed: ProblemEnum = Field(..., description="Main problem the startup addresses")
    
    # Founder Information (0-10 scale)
    founder_education_score: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Founder education background score (0-10)"
    )
    founder_experience_years: float = Field(
        ..., 
        ge=0, 
        le=20, 
        description="Years of relevant founder experience (0-20)"
    )
    founder_network_score: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Founder network strength score (0-10)"
    )
    
    # Article & Media Features
    article_mentions: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Number of article mentions"
    )
    keyword_relevance_score: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Keyword relevance score from articles (0-10)"
    )
    sentiment_score: float = Field(
        ..., 
        ge=-1, 
        le=1, 
        description="Sentiment score from media coverage (-1 to 1)"
    )
    
    # Customer & Market Features
    customer_testimonials: int = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Number of customer testimonials"
    )
    user_satisfaction_score: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="User satisfaction score (0-10)"
    )
    market_size_estimate: float = Field(
        ..., 
        ge=1000, 
        le=1000000000, 
        description="Market size estimate in USD"
    )
    problem_country_alignment: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="How well the problem aligns with country needs (0-10)"
    )
    
    # Operational Features
    months_in_operation: int = Field(
        ..., 
        ge=1, 
        le=120, 
        description="Months the startup has been operating (1-120)"
    )
    team_size: int = Field(
        ..., 
        ge=1, 
        le=200, 
        description="Current team size (1-200)"
    )
    funding_raised: float = Field(
        ..., 
        ge=0, 
        le=100000000, 
        description="Total funding raised in USD"
    )
    
    # Social Media & PR
    social_media_followers: int = Field(
        ..., 
        ge=0, 
        le=1000000, 
        description="Total social media followers"
    )
    media_coverage_count: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Number of media coverage instances"
    )
    
    @validator('founder_education_score', 'founder_network_score', 'keyword_relevance_score', 
              'user_satisfaction_score', 'problem_country_alignment')
    def validate_score_ranges(cls, v):
        """Ensure scores are within valid ranges"""
        if not 0 <= v <= 10:
            raise ValueError('Score must be between 0 and 10')
        return round(v, 2)
    
    @validator('founder_experience_years')
    def validate_experience(cls, v):
        """Validate founder experience"""
        if not 0 <= v <= 20:
            raise ValueError('Founder experience must be between 0 and 20 years')
        return round(v, 1)
    
    @validator('sentiment_score')
    def validate_sentiment(cls, v):
        """Validate sentiment score"""
        if not -1 <= v <= 1:
            raise ValueError('Sentiment score must be between -1 and 1')
        return round(v, 3)
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    """Model for prediction response"""
    
    investor_readiness_score: float = Field(..., description="Predicted investor readiness score (0-100)")
    confidence_lower: float = Field(..., description="Lower bound of confidence interval")
    confidence_upper: float = Field(..., description="Upper bound of confidence interval")
    risk_level: str = Field(..., description="Investment risk level")
    recommendation: str = Field(..., description="Investment recommendation")
    
    class Config:
        schema_extra = {
            "example": {
                "investor_readiness_score": 78.45,
                "confidence_lower": 73.45,
                "confidence_upper": 83.45,
                "risk_level": "Medium",
                "recommendation": "Promising startup with strong fundamentals. Consider for investment."
            }
        }

class BatchPredictionRequest(BaseModel):
    """Model for batch prediction requests"""
    startups: list[StartupPredictionRequest] = Field(
        ..., 
        max_items=10,
        description="List of startups to evaluate (max 10)"
    )

class BatchPredictionResponse(BaseModel):
    """Model for batch prediction responses"""
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    summary: dict = Field(..., description="Summary statistics")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    timestamp: str = Field(..., description="Response timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")

class StartupRecommendation(BaseModel):
    """Model for startup recommendation in lists"""
    startup_id: str = Field(..., description="Unique startup identifier")
    startup_name: str = Field(..., description="Company name")
    country: str = Field(..., description="Country of operation")
    sector: str = Field(..., description="Business sector")
    problem_addressed: str = Field(..., description="Main problem being solved")
    investor_readiness_score: float = Field(..., description="Predicted investment readiness (0-100)")
    funding_raised: float = Field(..., description="Total funding raised (USD)")
    team_size: int = Field(..., description="Current team size")
    months_in_operation: int = Field(..., description="Months in operation")
    problem_country_alignment: float = Field(..., description="How well problem aligns with country needs (0-10)")
    user_satisfaction_score: float = Field(..., description="Customer satisfaction score (0-10)")
    valuation_usd: Optional[float] = Field(None, description="Company valuation in USD")
    recent_milestone: Optional[str] = Field(None, description="Recent company milestone")
    
    class Config:
        schema_extra = {
            "example": {
                "startup_id": "REAL_FLUTTERWAVE",
                "startup_name": "Flutterwave",
                "country": "Nigeria",
                "sector": "FinTech",
                "problem_addressed": "Digital Payment",
                "investor_readiness_score": 95.5,
                "funding_raised": 500000000,
                "team_size": 260,
                "months_in_operation": 108,
                "problem_country_alignment": 9.2,
                "user_satisfaction_score": 8.7,
                "valuation_usd": 3000000000,
                "recent_milestone": "Launch in Zambia, seeking NGX listing"
            }
        }

class CountryRecommendationsResponse(BaseModel):
    """Response model for country-based recommendations"""
    country: str = Field(..., description="Country name")
    total_startups: int = Field(..., description="Total number of startups in country")
    recommended_startups: int = Field(..., description="Number of recommended startups (score >= 60)")
    average_readiness_score: float = Field(..., description="Average investor readiness score")
    top_sectors: list[str] = Field(..., description="Top 3 sectors in this country")
    startups: list[StartupRecommendation] = Field(..., description="List of recommended startups")

class SectorRecommendationsResponse(BaseModel):
    """Response model for sector-based recommendations"""
    sector: str = Field(..., description="Sector name")
    total_startups: int = Field(..., description="Total number of startups in sector")
    recommended_startups: int = Field(..., description="Number of recommended startups (score >= 60)")
    average_readiness_score: float = Field(..., description="Average investor readiness score")
    top_countries: list[str] = Field(..., description="Top 3 countries for this sector")
    startups: list[StartupRecommendation] = Field(..., description="List of recommended startups")

class CountrySectorRecommendationsResponse(BaseModel):
    """Response model for country + sector based recommendations"""
    country: str = Field(..., description="Country name")
    sector: str = Field(..., description="Sector name")
    total_startups: int = Field(..., description="Total number of startups in country+sector")
    average_readiness_score: float = Field(..., description="Average investor readiness score")
    startups: list[StartupRecommendation] = Field(..., description="List of recommended startups")

class TopCountriesResponse(BaseModel):
    """Response model for top investment countries"""
    country: str = Field(..., description="Country name")
    total_startups: int = Field(..., description="Total startups")
    recommended_startups: int = Field(..., description="Recommended startups (score >= 60)")
    average_readiness_score: float = Field(..., description="Average readiness score")
    total_funding_raised: float = Field(..., description="Total funding raised by all startups (USD)")
    top_sectors: list[str] = Field(..., description="Top sectors in this country")
    success_rate: float = Field(..., description="Percentage of startups with score >= 60")

class AfricaOverviewResponse(BaseModel):
    """Response model for Africa-wide investment overview"""
    total_startups: int = Field(..., description="Total startups across Africa")
    recommended_startups: int = Field(..., description="Total recommended startups (score >= 60)")
    countries_covered: int = Field(..., description="Number of countries represented")
    sectors_covered: int = Field(..., description="Number of sectors represented")
    total_funding: float = Field(..., description="Total funding across all startups (USD)")
    average_readiness_score: float = Field(..., description="Average readiness score across Africa")
    top_countries: list[TopCountriesResponse] = Field(..., description="Top 5 countries for investment")
    startups: list[StartupRecommendation] = Field(..., description="Top recommended startups across Africa")
