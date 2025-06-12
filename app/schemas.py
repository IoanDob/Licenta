from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict, Any

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    username: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    username: str
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class AnalysisCreate(BaseModel):
    filename: str
    model_used: str

class AnalysisResponse(BaseModel):
    id: int
    filename: str
    model_used: str
    total_transactions: int
    fraud_detected: int
    risk_score: float
    created_at: datetime
    

    class Config:
        from_attributes = True

class AnalysisResult(BaseModel):
    total_transactions: int
    fraud_detected: int
    fraud_percentage: float
    risk_score: float
    high_risk_transactions: List[Dict[str, Any]]  # Assuming transactions is a list of dictionaries
    summary_stats: Dict[str, Any]  # Assuming summary_stats is a dictionary with various statistics
    