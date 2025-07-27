from typing import Optional
from pydantic import BaseModel

# Define the request and response models
class PredictionRequest(BaseModel):
    country: Optional[str] = None
    sector: Optional[str] = None

class PredictionResponse(BaseModel):
    # Define the structure of your prediction response
    pass

def generate_prediction(request: PredictionRequest) -> PredictionResponse:
    """Generate predictions based on country and/or sector only."""
    
    filters = {}
    if request.country:
        filters['country'] = request.country
    if request.sector:
        filters['sector'] = request.sector
    
    # Apply filters and generate predictions
    # ...existing prediction logic with updated filters...
    
    return PredictionResponse(
        # ...existing response structure...
    )