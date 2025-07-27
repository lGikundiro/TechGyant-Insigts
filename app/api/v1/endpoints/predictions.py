from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.v1.schemas import PredictionRequest, PredictionResponse
from app.db.session import get_db

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def create_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Create predictions based on country and/or sector.
    
    Request can contain:
    - Only country
    - Only sector  
    - Both country and sector
    """
    # Validate that at least one field is provided
    if not request.country and not request.sector:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'country' or 'sector' must be provided"
        )
    
    # ...existing code for prediction logic...