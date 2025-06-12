# app/api/analysis.py - CORRECTED IMPORTS
"""
Simplified Analysis API with unified processing - FIXED VERSION
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any
import pandas as pd
import json
import logging
from datetime import datetime
import io

from ..database import get_db
from ..models import User, Analysis
from ..auth import get_current_user
from ..ml_models.model_utils import predictor

# CORRECTED IMPORT - use create_processor function
from ..utils.unified_processor import create_processor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.get("/test")
def test_analysis_api():
    """Test endpoint to verify analysis API functionality"""
    try:
        # Test unified processor
        processor = create_processor()
        processor_status = "✅ Ready"
        
        # Test ML models
        models_status = "✅ Loaded" if predictor.is_loaded else "⚠️ Not Loaded"
        
        return {
            "message": "Simplified Analysis API is working!",
            "status": "success",
            "components": {
                "unified_processor": processor_status,
                "ml_models": models_status
            },
            "features": [
                "Unified Column Detection",
                "Simplified Processing",
                "Multi-Model Analysis",
                "Real-time Insights"
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Analysis API test failed: {e}")
        raise HTTPException(status_code=500, detail=f"API test failed: {str(e)}")

@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Simplified upload and analysis endpoint
    """
    try:
        logger.info(f"Analysis request from user {current_user.username}")
        
        # Read file
        file_content = await file.read()
        df = await _read_csv_safe(file_content)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Could not read CSV file")
        
        # Process with unified processor
        processor = create_processor()
        processing_result = processor.process_file(df)
        
        if not processing_result['success']:
            raise HTTPException(status_code=400, detail="File processing failed")
        
        # Run ML prediction
        if not predictor.is_loaded:
            raise HTTPException(status_code=503, detail="ML models not loaded")
        
        paysim_df = processing_result['paysim_data']
        prediction_result = predictor.predict_fraud(paysim_df, model_type)
        
        # Combine results
        final_result = {
            **prediction_result,
            "file_info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns)
            },
            "detection_info": {
                "mapped_columns": len([r for r in processing_result['detection_results'].values() 
                                     if r.detected_type.value != 'other']),
                "quality_score": processing_result['quality_score'],
                "insights": processing_result['insights']
            },
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "processor_version": "2.0.0",
                "auto_processed": True
            }
        }
        
        # Save to database
        try:
            analysis_record = Analysis(
                user_id=current_user.id,
                filename=file.filename,
                model_used=model_type,
                total_transactions=prediction_result["total_transactions"],
                fraud_detected=prediction_result["fraud_detected"],
                risk_score=prediction_result["risk_score"],
                analysis_results=json.dumps(final_result, default=str)
            )
            
            db.add(analysis_record)
            db.commit()
            db.refresh(analysis_record)
            
            final_result["database_id"] = analysis_record.id
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            final_result["database_save_failed"] = True
        
        logger.info("Analysis completed successfully")
        return final_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/preview")
async def preview_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Preview file structure and column detection
    """
    try:
        file_content = await file.read()
        df = await _read_csv_safe(file_content)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Could not read CSV file")
        
        # Run detection only
        processor = create_processor()
        detection_results = processor.detect_columns(df)
        
        # Create preview
        preview_data = {
            "file_info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "size_mb": len(file_content) / 1024 / 1024
            },
            "column_detection": {
                col_name: {
                    "detected_type": result.detected_type.value,
                    "confidence": result.confidence,
                    "sample_values": result.sample_values[:3]
                }
                for col_name, result in detection_results.items()
            },
            "data_preview": df.head(5).to_dict('records'),
            "recommendations": processor.generate_insights(detection_results, df)
        }
        
        return preview_data
        
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _read_csv_safe(file_content: bytes) -> pd.DataFrame:
    """Safely read CSV with encoding detection"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    separators = [',', ';', '\t', '|']
    
    for encoding in encodings:
        for separator in separators:
            try:
                text_content = file_content.decode(encoding)
                df = pd.read_csv(
                    io.StringIO(text_content),
                    sep=separator,
                    low_memory=False
                )
                
                # Validate DataFrame
                if not df.empty and len(df.columns) >= 2 and len(df) >= 5:
                    return df
                    
            except Exception:
                continue
    
    return None