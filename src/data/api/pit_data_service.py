"""
Point-in-Time Data Service API.

This module provides REST API endpoints for accessing temporal data with 
comprehensive look-ahead bias prevention and high-performance querying.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import json
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Internal imports
from ..core.temporal import TemporalValue, TemporalStore, DataType
from ..models.taiwan_market import TaiwanTradingCalendar, create_taiwan_trading_calendar
from ..pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, PITResult, QueryMode, BiasCheckLevel
)
from ..pipeline.incremental_updater import IncrementalUpdater, UpdateRequest, UpdateMode
from ..ingestion.finlab_connector import FinLabConnector

logger = logging.getLogger(__name__)


# Pydantic models for API

class DataTypeEnum(str, Enum):
    """API data type enumeration."""
    PRICE = "price"
    VOLUME = "volume"
    FUNDAMENTAL = "fundamental"
    CORPORATE_ACTION = "corporate_action"
    MARKET_DATA = "market_data"
    NEWS = "news"
    TECHNICAL = "technical"


class QueryModeEnum(str, Enum):
    """API query mode enumeration."""
    STRICT = "strict"
    FAST = "fast"
    BULK = "bulk"
    REALTIME = "realtime"


class BiasCheckEnum(str, Enum):
    """API bias check level enumeration."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class PITQueryRequest(BaseModel):
    """Point-in-time query request model."""
    symbols: List[str] = Field(..., min_items=1, max_items=100)
    as_of_date: date = Field(...)
    data_types: List[DataTypeEnum] = Field(..., min_items=1)
    mode: QueryModeEnum = QueryModeEnum.STRICT
    bias_check: BiasCheckEnum = BiasCheckEnum.STRICT
    max_lag_days: Optional[int] = Field(None, ge=0, le=365)
    include_metadata: bool = True
    
    @validator('as_of_date')
    def validate_as_of_date(cls, v):
        if v > date.today():
            raise ValueError('as_of_date cannot be in the future')
        return v
    
    @validator('symbols')
    def validate_symbols(cls, v):
        # Basic Taiwan stock symbol validation
        for symbol in v:
            if not symbol.isdigit() or len(symbol) != 4:
                raise ValueError(f'Invalid Taiwan stock symbol: {symbol}')
        return v


class BulkQueryRequest(BaseModel):
    """Bulk query request model."""
    symbols: List[str] = Field(..., min_items=1, max_items=50)
    start_date: date = Field(...)
    end_date: date = Field(...)
    data_types: List[DataTypeEnum] = Field(..., min_items=1)
    mode: QueryModeEnum = QueryModeEnum.BULK
    bias_check: BiasCheckEnum = BiasCheckEnum.BASIC
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        if v > date.today():
            raise ValueError('end_date cannot be in the future')
        return v


class TemporalValueResponse(BaseModel):
    """Temporal value response model."""
    value: Union[float, int, str, Dict[str, Any]]
    as_of_date: date
    value_date: date
    data_type: DataTypeEnum
    symbol: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    version: int = 1


class PITQueryResponse(BaseModel):
    """Point-in-time query response model."""
    success: bool
    data: Dict[str, Dict[DataTypeEnum, TemporalValueResponse]]
    execution_time_ms: float
    cache_hit_rate: float = 0.0
    bias_violations: List[str] = []
    warnings: List[str] = []
    total_queries: int = 0


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    pit_engine_stats: Dict[str, Any]
    database_status: str


class TradingCalendarResponse(BaseModel):
    """Trading calendar response model."""
    date: date
    is_trading_day: bool
    market_session: str
    trading_hours: Optional[Tuple[str, str]] = None
    notes: Optional[str] = None


class DataQualityRequest(BaseModel):
    """Data quality validation request."""
    symbol: str = Field(..., regex=r'^\d{4}$')
    data_date: date
    data: Dict[str, Any]


class DataQualityResponse(BaseModel):
    """Data quality validation response."""
    symbol: str
    data_date: date
    is_valid: bool
    issues: List[str] = []
    warnings: List[str] = []


# Service class

class PITDataService:
    """Point-in-time data service with REST API."""
    
    def __init__(self,
                 temporal_store: TemporalStore,
                 pit_engine: PointInTimeEngine,
                 incremental_updater: Optional[IncrementalUpdater] = None,
                 finlab_connector: Optional[FinLabConnector] = None,
                 enable_background_updates: bool = True):
        
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        self.incremental_updater = incremental_updater
        self.finlab_connector = finlab_connector
        self.enable_background_updates = enable_background_updates
        
        # Trading calendar
        self.trading_calendar = create_taiwan_trading_calendar(2024)
        
        # Service state
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        
        # Background task executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("PIT Data Service initialized")
    
    def _convert_data_types(self, api_types: List[DataTypeEnum]) -> List[DataType]:
        """Convert API data types to internal enum."""
        mapping = {
            DataTypeEnum.PRICE: DataType.PRICE,
            DataTypeEnum.VOLUME: DataType.VOLUME,
            DataTypeEnum.FUNDAMENTAL: DataType.FUNDAMENTAL,
            DataTypeEnum.CORPORATE_ACTION: DataType.CORPORATE_ACTION,
            DataTypeEnum.MARKET_DATA: DataType.MARKET_DATA,
            DataTypeEnum.NEWS: DataType.NEWS,
            DataTypeEnum.TECHNICAL: DataType.TECHNICAL
        }
        return [mapping[dt] for dt in api_types]
    
    def _convert_query_mode(self, api_mode: QueryModeEnum) -> QueryMode:
        """Convert API query mode to internal enum."""
        mapping = {
            QueryModeEnum.STRICT: QueryMode.STRICT,
            QueryModeEnum.FAST: QueryMode.FAST,
            QueryModeEnum.BULK: QueryMode.BULK,
            QueryModeEnum.REALTIME: QueryMode.REALTIME
        }
        return mapping[api_mode]
    
    def _convert_bias_check(self, api_bias: BiasCheckEnum) -> BiasCheckLevel:
        """Convert API bias check to internal enum."""
        mapping = {
            BiasCheckEnum.NONE: BiasCheckLevel.NONE,
            BiasCheckEnum.BASIC: BiasCheckLevel.BASIC,
            BiasCheckEnum.STRICT: BiasCheckLevel.STRICT,
            BiasCheckEnum.PARANOID: BiasCheckLevel.PARANOID
        }
        return mapping[api_bias]
    
    def _temporal_value_to_response(self, value: TemporalValue) -> TemporalValueResponse:
        """Convert internal temporal value to API response."""
        # Convert Decimal to float for JSON serialization
        api_value = value.value
        if isinstance(api_value, Decimal):
            api_value = float(api_value)
        
        return TemporalValueResponse(
            value=api_value,
            as_of_date=value.as_of_date,
            value_date=value.value_date,
            data_type=DataTypeEnum(value.data_type.value),
            symbol=value.symbol,
            metadata=value.metadata,
            created_at=value.created_at,
            version=value.version
        )
    
    async def execute_pit_query(self, request: PITQueryRequest) -> PITQueryResponse:
        """Execute point-in-time query."""
        try:
            self.request_count += 1
            
            # Convert API request to internal query
            internal_query = PITQuery(
                symbols=request.symbols,
                as_of_date=request.as_of_date,
                data_types=self._convert_data_types(request.data_types),
                mode=self._convert_query_mode(request.mode),
                bias_check=self._convert_bias_check(request.bias_check),
                max_lag_days=request.max_lag_days,
                include_metadata=request.include_metadata
            )
            
            # Execute query
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.pit_engine.execute_query,
                internal_query
            )
            
            # Convert result to API response
            response_data = {}
            for symbol, symbol_data in result.data.items():
                response_data[symbol] = {}
                for data_type, temporal_value in symbol_data.items():
                    api_data_type = DataTypeEnum(data_type.value)
                    response_data[symbol][api_data_type] = self._temporal_value_to_response(temporal_value)
            
            return PITQueryResponse(
                success=True,
                data=response_data,
                execution_time_ms=result.execution_time_ms,
                cache_hit_rate=result.cache_hit_rate,
                bias_violations=result.bias_violations,
                warnings=result.warnings,
                total_queries=result.total_queries
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"PIT query execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def execute_bulk_query(self, request: BulkQueryRequest) -> List[PITQueryResponse]:
        """Execute bulk historical query."""
        try:
            responses = []
            current_date = request.start_date
            
            while current_date <= request.end_date:
                # Create point-in-time query for each date
                pit_request = PITQueryRequest(
                    symbols=request.symbols,
                    as_of_date=current_date,
                    data_types=request.data_types,
                    mode=request.mode,
                    bias_check=request.bias_check,
                    include_metadata=False  # Reduce payload for bulk
                )
                
                response = await self.execute_pit_query(pit_request)
                responses.append(response)
                
                current_date += timedelta(days=1)
            
            return responses
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Bulk query execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def validate_data_quality(self, request: DataQualityRequest) -> DataQualityResponse:
        """Validate data quality for a symbol."""
        try:
            if self.finlab_connector:
                issues = self.finlab_connector.validate_data_quality(
                    request.symbol,
                    request.data_date,
                    request.data
                )
                
                return DataQualityResponse(
                    symbol=request.symbol,
                    data_date=request.data_date,
                    is_valid=len(issues) == 0,
                    issues=issues,
                    warnings=[]
                )
            else:
                return DataQualityResponse(
                    symbol=request.symbol,
                    data_date=request.data_date,
                    is_valid=True,
                    issues=[],
                    warnings=["Data quality validation not available - no connector"]
                )
                
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_trading_calendar(self, 
                                  start_date: date,
                                  end_date: date) -> List[TradingCalendarResponse]:
        """Get trading calendar information."""
        try:
            calendar_entries = []
            current_date = start_date
            
            while current_date <= end_date:
                calendar_entry = self.trading_calendar.get(current_date)
                
                if calendar_entry:
                    trading_hours = None
                    if calendar_entry.is_trading_day:
                        hours = calendar_entry.trading_hours
                        if hours[0] and hours[1]:
                            trading_hours = (
                                hours[0].strftime("%H:%M"),
                                hours[1].strftime("%H:%M")
                            )
                    
                    calendar_entries.append(TradingCalendarResponse(
                        date=current_date,
                        is_trading_day=calendar_entry.is_trading_day,
                        market_session=calendar_entry.market_session.value,
                        trading_hours=trading_hours,
                        notes=calendar_entry.notes
                    ))
                
                current_date += timedelta(days=1)
            
            return calendar_entries
            
        except Exception as e:
            logger.error(f"Trading calendar query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_health(self) -> HealthResponse:
        """Get service health status."""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            pit_stats = self.pit_engine.get_performance_stats()
            
            # Check database connectivity
            database_status = "connected"
            if self.finlab_connector:
                try:
                    # Simple connectivity test
                    symbols = self.finlab_connector.get_available_symbols()
                    if not symbols:
                        database_status = "no_data"
                except Exception:
                    database_status = "disconnected"
            else:
                database_status = "not_configured"
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow(),
                version="1.0.0",
                uptime_seconds=uptime,
                pit_engine_stats=pit_stats,
                database_status=database_status
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow(),
                version="1.0.0",
                uptime_seconds=0,
                pit_engine_stats={},
                database_status="error"
            )
    
    async def trigger_background_update(self, background_tasks: BackgroundTasks):
        """Trigger background data update."""
        if self.incremental_updater and self.enable_background_updates:
            
            def run_update():
                # Create daily update request for major Taiwan stocks
                symbols = ["2330", "2317", "2454", "2881", "6505"]
                update_request = self.incremental_updater.create_daily_update_request(symbols)
                result = self.incremental_updater.execute_update(update_request)
                logger.info(f"Background update completed: {result.total_changes} changes")
            
            background_tasks.add_task(run_update)
            return {"message": "Background update triggered"}
        else:
            return {"message": "Background updates not available"}


# FastAPI application factory

def create_api_app(pit_data_service: PITDataService) -> FastAPI:
    """Create FastAPI application with PIT data endpoints."""
    
    app = FastAPI(
        title="ML4T Point-in-Time Data API",
        description="High-performance API for temporal financial data with look-ahead bias prevention",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for security
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency for service injection
    def get_pit_service():
        return pit_data_service
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check(service: PITDataService = Depends(get_pit_service)):
        """Service health check."""
        return await service.get_health()
    
    # Point-in-time data endpoint
    @app.post("/data/pit", response_model=PITQueryResponse)
    async def get_pit_data(
        request: PITQueryRequest,
        service: PITDataService = Depends(get_pit_service)
    ):
        """Get point-in-time data for symbols."""
        return await service.execute_pit_query(request)
    
    # Convenience endpoint for single symbol
    @app.get("/data/pit/{symbol}", response_model=PITQueryResponse)
    async def get_symbol_pit_data(
        symbol: str = Path(..., regex=r'^\d{4}$'),
        as_of_date: date = Query(...),
        data_types: List[DataTypeEnum] = Query(...),
        mode: QueryModeEnum = Query(QueryModeEnum.STRICT),
        bias_check: BiasCheckEnum = Query(BiasCheckEnum.STRICT),
        service: PITDataService = Depends(get_pit_service)
    ):
        """Get point-in-time data for a single symbol."""
        request = PITQueryRequest(
            symbols=[symbol],
            as_of_date=as_of_date,
            data_types=data_types,
            mode=mode,
            bias_check=bias_check
        )
        return await service.execute_pit_query(request)
    
    # Bulk historical data endpoint
    @app.post("/data/bulk", response_model=List[PITQueryResponse])
    async def get_bulk_data(
        request: BulkQueryRequest,
        service: PITDataService = Depends(get_pit_service)
    ):
        """Get bulk historical data."""
        return await service.execute_bulk_query(request)
    
    # Data quality validation endpoint
    @app.post("/data/quality/validate", response_model=DataQualityResponse)
    async def validate_data_quality(
        request: DataQualityRequest,
        service: PITDataService = Depends(get_pit_service)
    ):
        """Validate data quality for a symbol."""
        return await service.validate_data_quality(request)
    
    # Trading calendar endpoint
    @app.get("/data/calendar/trading/{start_date}/{end_date}", 
             response_model=List[TradingCalendarResponse])
    async def get_trading_calendar(
        start_date: date = Path(...),
        end_date: date = Path(...),
        service: PITDataService = Depends(get_pit_service)
    ):
        """Get Taiwan trading calendar."""
        return await service.get_trading_calendar(start_date, end_date)
    
    # Background update trigger
    @app.post("/admin/update")
    async def trigger_update(
        background_tasks: BackgroundTasks,
        service: PITDataService = Depends(get_pit_service)
    ):
        """Trigger background data update."""
        return await service.trigger_background_update(background_tasks)
    
    # Statistics endpoint
    @app.get("/admin/stats")
    async def get_statistics(service: PITDataService = Depends(get_pit_service)):
        """Get service statistics."""
        pit_stats = service.pit_engine.get_performance_stats()
        
        service_stats = {
            "request_count": service.request_count,
            "error_count": service.error_count,
            "error_rate": service.error_count / max(service.request_count, 1),
            "uptime_seconds": (datetime.utcnow() - service.start_time).total_seconds()
        }
        
        if service.incremental_updater:
            updater_stats = service.incremental_updater.get_performance_stats()
            service_stats["updater_stats"] = updater_stats
        
        if service.finlab_connector:
            connector_stats = service.finlab_connector.get_performance_stats()
            service_stats["connector_stats"] = connector_stats
        
        return {
            "service": service_stats,
            "pit_engine": pit_stats
        }
    
    return app


# CLI for running the service

def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    temporal_store: Optional[TemporalStore] = None,
    finlab_config: Optional[Dict[str, Any]] = None
):
    """Run the PIT Data API server."""
    
    # Initialize components
    if temporal_store is None:
        from ..core.temporal import InMemoryTemporalStore
        temporal_store = InMemoryTemporalStore()
    
    # Create PIT engine
    pit_engine = PointInTimeEngine(temporal_store)
    
    # Create FinLab connector if config provided
    finlab_connector = None
    if finlab_config:
        from ..ingestion.finlab_connector import create_finlab_connector
        finlab_connector = create_finlab_connector(
            temporal_store=temporal_store,
            **finlab_config
        )
        finlab_connector.connect()
    
    # Create incremental updater
    incremental_updater = None
    if finlab_connector:
        incremental_updater = IncrementalUpdater(
            temporal_store, finlab_connector
        )
        incremental_updater.start_background_processing()
    
    # Create service and app
    service = PITDataService(
        temporal_store=temporal_store,
        pit_engine=pit_engine,
        incremental_updater=incremental_updater,
        finlab_connector=finlab_connector
    )
    
    app = create_api_app(service)
    
    # Run server
    logger.info(f"Starting PIT Data API server on {host}:{port}")
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        # Cleanup
        if incremental_updater:
            incremental_updater.stop_background_processing()
        if finlab_connector:
            finlab_connector.disconnect()


if __name__ == "__main__":
    # Example configuration
    finlab_config = {
        "host": "localhost",
        "port": 5432,
        "database": "finlab",
        "username": "finlab",
        "password": "password"
    }
    
    run_api_server(finlab_config=finlab_config)