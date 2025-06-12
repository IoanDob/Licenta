# app/main.py - FIXED VERSION
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import logging
import os
from contextlib import asynccontextmanager

# Import routers
from .api import auth as auth_router
from .api import analysis as analysis_router  
from .api import history as history_router

# Import database and models
from .database import engine, get_db
from .models import Base
from .auth import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Smart Detection starting up...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("📊 Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
    
    # Initialize ML models
    try:
        from .ml_models.model_utils import predictor
        if predictor.load_models():
            logger.info("🤖 ML models loaded successfully")
        else:
            logger.warning("⚠️ ML models not found - training may be required")
    except Exception as e:
        logger.error(f"❌ ML model initialization failed: {e}")
    
    # Test unified processor (optional - just for verification)
    try:
        from .utils.unified_processor import create_processor
        processor = create_processor()
        logger.info("🧠 Unified processor initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Unified processor test failed: {e}")
        # This is not critical - the app can still work
    
    logger.info("✅ Smart Detection startup complete")
    
    yield
    
    logger.info("🛑 Smart Detection shutting down...")

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Smart Detection",
    description="AI-Powered Fraud Detection with Smart Column Detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with enhanced configuration
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Enhanced templates configuration
templates = Jinja2Templates(directory="frontend/templates")

# Include API routers with enhanced configuration
app.include_router(
    auth_router.router,
    prefix="/api",
    tags=["Authentication"]
)

app.include_router(
    analysis_router.router,
    prefix="/api",
    tags=["Analysis"]
)

app.include_router(
    history_router.router,
    prefix="/api", 
    tags=["History"]
)

# Enhanced route handlers
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def login_page(request: Request):
    """Enhanced login page with improved styling"""
    return templates.TemplateResponse(
        "login.html", 
        {"request": request}
    )

@app.get("/register", response_class=HTMLResponse, tags=["Frontend"])
async def register_page(request: Request):
    """Enhanced registration page"""
    return templates.TemplateResponse(
        "register.html",
        {"request": request}
    )

@app.get("/dashboard", response_class=HTMLResponse, tags=["Frontend"])
async def dashboard_page(request: Request):
    """Enhanced dashboard with smart detection features"""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )

@app.get("/analysis", response_class=HTMLResponse, tags=["Frontend"])
async def analysis_page(request: Request):
    """Enhanced analysis page with smart column detection"""
    return templates.TemplateResponse(
        "analysis.html",
        {"request": request}
    )

@app.get("/history", response_class=HTMLResponse, tags=["Frontend"])
async def history_page(request: Request):
    """Enhanced history page with detailed analysis views"""
    return templates.TemplateResponse(
        "history.html",
        {"request": request}
    )

# System status and health endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """Enhanced health check with component status"""
    try:
        # Check database
        db_status = "healthy"
        try:
            from .database import SessionLocal
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Check ML models
        ml_status = "healthy"
        try:
            from .ml_models.model_utils import predictor
            if not predictor.is_loaded:
                ml_status = "models not loaded"
        except Exception as e:
            ml_status = f"unhealthy: {str(e)}"
        
        # Check unified processor
        processor_status = "healthy"
        try:
            from .utils.unified_processor import create_processor
            processor = create_processor()
        except Exception as e:
            processor_status = f"warning: {str(e)}"
        
        return {
            "status": "healthy" if all("healthy" in s for s in [db_status, ml_status]) else "degraded",
            "components": {
                "database": db_status,
                "ml_models": ml_status,
                "unified_processor": processor_status
            },
            "version": "2.0.0",
            "features": [
                "Unified Column Detection",
                "Simplified Processing", 
                "Multi-Model Analysis",
                "Real-time Processing"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/system/capabilities", tags=["System"])
async def system_capabilities():
    """Get comprehensive system capabilities"""
    try:
        from .utils.unified_processor import ColumnType
        from .ml_models.model_utils import predictor
        
        capabilities = {
            "unified_processing": {
                "supported_column_types": [col_type.value for col_type in ColumnType],
                "auto_mapping": True,
                "confidence_scoring": True,
                "conflict_resolution": True
            },
            "file_processing": {
                "supported_formats": [".csv"],
                "encoding_detection": True,
                "separator_detection": True,
                "max_file_size_mb": 500,
                "paysim_conversion": True
            },
            "ml_models": {
                "available_models": ["random_forest", "logistic_regression"],
                "models_loaded": predictor.is_loaded if 'predictor' in locals() else False,
                "feature_engineering": True,
                "real_time_prediction": True
            },
            "analysis_types": [
                "Basic Fraud Detection",
                "Enhanced Fraud Detection", 
                "Network Analysis",
                "Temporal Analysis"
            ],
            "api_features": {
                "unified_processing": True,
                "simplified_endpoints": True,
                "batch_processing": True,
                "preview_mode": True
            }
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        return {"error": str(e)}

# Enhanced error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Enhanced 404 handler"""
    return templates.TemplateResponse(
        "login.html",
        {"request": request},
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Enhanced 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

# Development helpers
@app.get("/dev/reset-demo", tags=["Development"])
async def reset_demo_data(db: Session = Depends(get_db)):
    """Reset demo data for development (remove in production)"""
    if os.getenv("DEBUG", "False").lower() == "true":
        try:
            # Clear existing demo data
            from .models import Analysis
            db.query(Analysis).delete()
            db.commit()
            
            logger.info("Demo data reset completed")
            return {"message": "Demo data reset successfully"}
        except Exception as e:
            logger.error(f"Demo reset failed: {e}")
            return {"error": str(e)}
    else:
        raise HTTPException(status_code=403, detail="Only available in debug mode")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )