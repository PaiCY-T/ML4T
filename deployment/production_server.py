"""
Production Server for ML4T LightGBM Inference System.

FastAPI-based production server providing:
- Real-time inference API endpoints
- Health monitoring and status checks
- Performance metrics collection
- Graceful startup and shutdown
- Taiwan market specific features
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.lightgbm_alpha import LightGBMAlphaModel, ModelConfig
from src.inference.realtime import RealtimePredictor, InferenceConfig
from src.monitoring.model_health import ModelHealthMonitor, MonitoringConfig
from src.factors.base import FactorEngine

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Pydantic models for API
class PredictionRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to predict")
    timestamp: Optional[str] = Field(None, description="Timestamp for prediction (ISO format)")
    features: Optional[Dict[str, List[float]]] = Field(None, description="Optional precomputed features")


class PredictionResponse(BaseModel):
    predictions: Dict[str, float] = Field(..., description="Symbol to prediction mapping")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    success: bool = Field(True, description="Request success status")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime")
    inference_ready: bool = Field(..., description="Inference system ready")
    model_loaded: bool = Field(..., description="Model loaded successfully")
    health_score: float = Field(..., description="Overall health score 0-100")
    details: Dict = Field(..., description="Detailed health metrics")


class MetricsResponse(BaseModel):
    inference_metrics: Dict = Field(..., description="Inference performance metrics")
    health_metrics: Dict = Field(..., description="Model health metrics")
    system_metrics: Dict = Field(..., description="System resource metrics")


# Global application state
class AppState:
    def __init__(self):
        self.model: Optional[LightGBMAlphaModel] = None
        self.predictor: Optional[RealtimePredictor] = None
        self.monitor: Optional[ModelHealthMonitor] = None
        self.factor_engine: Optional[FactorEngine] = None
        self.startup_time = datetime.now()
        self.ready = False
        self.shutdown_event = asyncio.Event()


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting ML4T Inference Server...")
    
    try:
        await startup_system()
        logger.info("ML4T Inference Server started successfully")
        yield
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ML4T Inference Server...")
        await shutdown_system()
        logger.info("ML4T Inference Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ML4T LightGBM Inference API",
    description="Production inference system for Taiwan equity alpha model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def startup_system():
    """Initialize the inference system."""
    try:
        # Load configuration from environment
        model_path = os.getenv('MODEL_PATH', '/app/models/lightgbm_alpha.pkl')
        max_workers = int(os.getenv('MAX_WORKERS', '4'))
        batch_size = int(os.getenv('BATCH_SIZE', '500'))
        max_latency_ms = float(os.getenv('MAX_LATENCY_MS', '100'))
        enable_gpu = os.getenv('ENABLE_GPU', 'false').lower() == 'true'
        
        logger.info("Loading configuration", 
                   model_path=model_path,
                   max_workers=max_workers,
                   batch_size=batch_size,
                   max_latency_ms=max_latency_ms,
                   enable_gpu=enable_gpu)
        
        # Initialize model
        logger.info("Loading LightGBM model...")
        app_state.model = LightGBMAlphaModel()
        
        if Path(model_path).exists():
            app_state.model.load_model(model_path)
            logger.info("Model loaded successfully", path=model_path)
        else:
            logger.warning("Model file not found, using untrained model", path=model_path)
        
        # Initialize factor engine (mock for now)
        logger.info("Initializing factor engine...")
        app_state.factor_engine = MockFactorEngine()
        
        # Initialize inference system
        logger.info("Initializing inference system...")
        inference_config = InferenceConfig(
            max_latency_ms=max_latency_ms,
            batch_size=batch_size,
            max_workers=max_workers,
            enable_gpu=enable_gpu
        )
        
        app_state.predictor = RealtimePredictor(
            model=app_state.model,
            factor_engine=app_state.factor_engine,
            config=inference_config
        )
        
        # Warmup the system
        logger.info("Warming up inference system...")
        warmup_metrics = app_state.predictor.warmup(n_symbols=100)
        logger.info("Warmup completed", metrics=warmup_metrics)
        
        # Initialize monitoring
        logger.info("Initializing monitoring system...")
        monitoring_config = MonitoringConfig(
            enable_email_alerts=os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
            alert_recipients=os.getenv('ALERT_RECIPIENTS', '').split(',') if os.getenv('ALERT_RECIPIENTS') else []
        )
        
        app_state.monitor = ModelHealthMonitor(
            model=app_state.model,
            predictor=app_state.predictor,
            config=monitoring_config
        )
        
        app_state.monitor.start_monitoring()
        
        # Start background monitoring task
        asyncio.create_task(monitoring_background_task())
        
        app_state.ready = True
        logger.info("System startup completed successfully")
        
    except Exception as e:
        logger.error("System startup failed", error=str(e), exc_info=True)
        raise


async def shutdown_system():
    """Cleanup system resources."""
    try:
        app_state.shutdown_event.set()
        
        if app_state.monitor:
            app_state.monitor.stop_monitoring()
            
        app_state.ready = False
        logger.info("System shutdown completed")
        
    except Exception as e:
        logger.error("Shutdown error", error=str(e), exc_info=True)


async def monitoring_background_task():
    """Background task for health monitoring."""
    while not app_state.shutdown_event.is_set():
        try:
            if app_state.monitor and app_state.ready:
                # Update health metrics
                health_metrics = app_state.monitor.update_health()
                
                if health_metrics.overall_status.value == 'critical':
                    logger.error("System health critical", 
                               health_score=health_metrics.get_health_score())
                
            await asyncio.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error("Monitoring background task error", error=str(e))
            await asyncio.sleep(60)


def get_system_ready():
    """Dependency to check if system is ready."""
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not ready")
    return True


# Mock factor engine for demo purposes
class MockFactorEngine:
    """Mock factor engine for testing and demo."""
    
    def compute_factors_for_symbol(self, symbol: str, timestamp: datetime, feature_columns: List[str]):
        """Mock factor computation for single symbol."""
        import numpy as np
        import pandas as pd
        return pd.Series(np.random.randn(len(feature_columns)), index=feature_columns)
    
    def compute_factors_batch(self, symbols: List[str], timestamp: datetime, feature_columns: List[str]):
        """Mock factor computation for multiple symbols."""
        import numpy as np
        import pandas as pd
        return pd.DataFrame(
            np.random.randn(len(symbols), len(feature_columns)),
            index=symbols,
            columns=feature_columns
        )


# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "ML4T LightGBM Inference API",
        "version": "1.0.0",
        "status": "running" if app_state.ready else "starting",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    ready: bool = Depends(get_system_ready)
):
    """Generate predictions for given symbols."""
    try:
        start_time = datetime.now()
        
        # Parse timestamp if provided
        timestamp = datetime.fromisoformat(request.timestamp) if request.timestamp else start_time
        
        logger.info("Prediction request received", 
                   symbols_count=len(request.symbols),
                   timestamp=timestamp.isoformat())
        
        # Generate predictions
        predictions_series, latency_ms = app_state.predictor.predict_batch(
            symbols=request.symbols,
            timestamp=timestamp
        )
        
        # Convert to dictionary
        predictions_dict = predictions_series.to_dict()
        
        # Log performance
        logger.info("Prediction completed",
                   symbols_count=len(request.symbols),
                   latency_ms=latency_ms,
                   success=True)
        
        # Background health update
        background_tasks.add_task(update_monitoring_metrics, len(request.symbols), latency_ms)
        
        return PredictionResponse(
            predictions=predictions_dict,
            latency_ms=latency_ms,
            timestamp=timestamp.isoformat(),
            model_version="1.0.0",
            success=True
        )
        
    except Exception as e:
        logger.error("Prediction failed", 
                    error=str(e),
                    symbols_count=len(request.symbols) if request.symbols else 0,
                    exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check endpoint."""
    try:
        current_time = datetime.now()
        uptime_seconds = (current_time - app_state.startup_time).total_seconds()
        
        # Basic health checks
        model_loaded = app_state.model is not None and app_state.model.model is not None
        inference_ready = app_state.predictor is not None and app_state.ready
        
        # Get detailed health metrics
        health_details = {}
        health_score = 50.0  # Default neutral score
        
        if app_state.monitor:
            try:
                health_report = app_state.monitor.get_health_report()
                health_score = health_report.get('health_score', 50.0)
                health_details = health_report
            except Exception as e:
                logger.warning("Failed to get detailed health metrics", error=str(e))
        
        # Determine overall status
        if not inference_ready or not model_loaded:
            status = "unhealthy"
        elif health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        else:
            status = "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=current_time.isoformat(),
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            inference_ready=inference_ready,
            model_loaded=model_loaded,
            health_score=health_score,
            details=health_details
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e), exc_info=True)
        
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=0,
            inference_ready=False,
            model_loaded=False,
            health_score=0.0,
            details={"error": str(e)}
        )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics(ready: bool = Depends(get_system_ready)):
    """Get comprehensive system metrics."""
    try:
        # Get inference metrics
        inference_metrics = {}
        if app_state.predictor:
            inference_metrics = app_state.predictor.get_performance_summary()
        
        # Get health metrics
        health_metrics = {}
        if app_state.monitor:
            health_report = app_state.monitor.get_health_report()
            health_metrics = health_report.get('metrics', {})
        
        # Get system metrics
        import psutil
        process = psutil.Process()
        
        system_metrics = {
            'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'uptime_seconds': (datetime.now() - app_state.startup_time).total_seconds()
        }
        
        return MetricsResponse(
            inference_metrics=inference_metrics,
            health_metrics=health_metrics,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error("Metrics collection failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@app.get("/model/info")
async def model_info(ready: bool = Depends(get_system_ready)):
    """Get model information and metadata."""
    try:
        if not app_state.model or not app_state.model.model:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        model_summary = app_state.model.get_model_summary()
        
        return {
            "model_summary": model_summary,
            "feature_count": len(app_state.model.feature_columns),
            "model_type": "LightGBM",
            "version": "1.0.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model info retrieval failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model info failed: {str(e)}")


async def update_monitoring_metrics(symbol_count: int, latency_ms: float):
    """Background task to update monitoring metrics."""
    try:
        if app_state.monitor:
            # Update inference metrics
            app_state.monitor.current_health.avg_inference_latency_ms = latency_ms
            app_state.monitor.current_health.last_updated = datetime.now()
            
    except Exception as e:
        logger.warning("Monitoring metrics update failed", error=str(e))


# Signal handlers for graceful shutdown
def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal", signal=signum)
    app_state.shutdown_event.set()


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


if __name__ == "__main__":
    # Production server configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    workers = int(os.getenv('WORKERS', '1'))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    logger.info("Starting production server",
               host=host,
               port=port,
               workers=workers,
               log_level=log_level)
    
    # Run with uvicorn
    uvicorn.run(
        "production_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        reload=False
    )