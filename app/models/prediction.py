from pydantic import BaseModel, validator
from typing import Optional

class PredictionRequest(BaseModel):
    country: Optional[str] = None
    sector: Optional[str] = None
    
    @validator('country', 'sector')
    def validate_at_least_one_field(cls, v, values):
        if not v and not any(values.values()):
            raise ValueError('At least one of country or sector must be provided')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "country": "USA",
                "sector": "Technology"
            }
        }