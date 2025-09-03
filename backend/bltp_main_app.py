"""
BLTP Platform Backend - Indian Bond Trading Platform
Main Application Setup with FastAPI
"""

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import json
import logging
from datetime import datetime, timedelta
import redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from passlib.context import CryptContext
import jwt
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import httpx
import websockets
from decimal import Decimal
import pandas as pd
import numpy as np

# Configuration
DATABASE_URL = "postgresql://username:password@localhost/bltp_db"
REDIS_URL = "redis://localhost:6379"
JWT_SECRET_KEY = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"

# Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis Setup
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    kyc_status = Column(String, default="pending")  # pending, verified, rejected
    pan_number = Column(String, unique=True)
    phone = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolios = relationship("Portfolio", back_populates="user")
    trades = relationship("Trade", back_populates="user")

class Bond(Base):
    __tablename__ = "bonds"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    isin = Column(String, unique=True, nullable=False)  # International Securities Identification Number
    name = Column(String, nullable=False)
    issuer = Column(String, nullable=False)
    face_value = Column(Float, nullable=False)
    coupon_rate = Column(Float, nullable=False)
    maturity_date = Column(DateTime, nullable=False)
    issue_date = Column(DateTime, nullable=False)
    bond_type = Column(String, nullable=False)  # government, corporate, municipal
    rating = Column(String)  # AAA, AA+, etc.
    current_price = Column(Float)
    ytm = Column(Float)  # Yield to Maturity
    duration = Column(Float)
    is_tradable = Column(Boolean, default=True)
    min_investment = Column(Float, default=1000)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    trades = relationship("Trade", back_populates="bond")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    total_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio")

class Holding(Base):
    __tablename__ = "holdings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    bond_id = Column(UUID(as_uuid=True), ForeignKey("bonds.id"), nullable=False)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_value = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    
    portfolio = relationship("Portfolio", back_populates="holdings")
    bond = relationship("Bond")

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    bond_id = Column(UUID(as_uuid=True), ForeignKey("bonds.id"), nullable=False)
    trade_type = Column(String, nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    status = Column(String, default="pending")  # pending, executed, cancelled
    order_type = Column(String, default="market")  # market, limit
    executed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="trades")
    bond = relationship("Bond", back_populates="trades")

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, nullable=False)  # NIFTY, SENSEX, bond ISINs
    price = Column(Float, nullable=False)
    change = Column(Float)
    change_percent = Column(Float)
    volume = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    pan_number: str
    phone: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class BondResponse(BaseModel):
    id: str
    isin: str
    name: str
    issuer: str
    current_price: float
    ytm: float
    rating: str
    maturity_date: str

class TradeRequest(BaseModel):
    bond_id: str
    trade_type: str
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    timestamp: str

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Market Data Service
class MarketDataService:
    def __init__(self):
        self.nse_base_url = "https://www.nseindia.com/api"
        self.bse_base_url = "https://api.bseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }

    async def fetch_nse_indices(self):
        """Fetch NSE indices data"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.nse_base_url}/allIndices",
                    headers=self.headers,
                    timeout=10.0
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error fetching NSE data: {e}")
            return None

    async def fetch_bond_prices(self, isin_list: List[str]):
        """Fetch bond prices - placeholder for actual bond data API"""
        # In production, integrate with bond data providers like FIMMDA, CCIL
        bond_prices = {}
        for isin in isin_list:
            # Simulate real-time bond price
            base_price = redis_client.get(f"bond_price:{isin}")
            if base_price:
                price = float(base_price) + np.random.uniform(-0.5, 0.5)
            else:
                price = 100 + np.random.uniform(-5, 5)  # Par value ± variation
            
            bond_prices[isin] = {
                'price': round(price, 2),
                'timestamp': datetime.utcnow().isoformat(),
                'ytm': round(6.5 + np.random.uniform(-1, 1), 2)
            }
            
            # Cache the price
            redis_client.setex(f"bond_price:{isin}", 300, price)
        
        return bond_prices

market_service = MarketDataService()

# Background tasks for real-time data
async def market_data_updater():
    """Background task to update market data"""
    while True:
        try:
            # Fetch NSE indices
            indices_data = await market_service.fetch_nse_indices()
            if indices_data:
                await manager.broadcast(json.dumps({
                    'type': 'market_update',
                    'data': indices_data
                }))
            
            # Fetch bond prices for active bonds
            active_bonds = ["IN0123456789", "IN0987654321"]  # Sample ISINs
            bond_prices = await market_service.fetch_bond_prices(active_bonds)
            
            await manager.broadcast(json.dumps({
                'type': 'bond_update',
                'data': bond_prices
            }))
            
        except Exception as e:
            logger.error(f"Error in market data updater: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

# Utility functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting BLTP Backend...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Start background tasks
    asyncio.create_task(market_data_updater())
    
    yield
    
    # Shutdown
    logger.info("Shutting down BLTP Backend...")

# FastAPI App
app = FastAPI(
    title="BLTP - Bond Liquidity Trading Platform",
    description="Backend API for Indian Bond Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )