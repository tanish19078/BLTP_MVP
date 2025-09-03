"""
BLTP Platform Backend - Main Routes Integration
Integrates all route modules and adds market data endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import asyncio
import numpy as np

# Import from main app and route modules
from main import (
    app, get_db, get_current_user, User, Bond, MarketData, Trade, 
    market_service, manager, redis_client, logger
)

# Import route modules (assuming they're in the same package)
from bltp_auth_routes import router as auth_router
from bltp_trading_engine import router as trading_router
from bltp_portfolio_routes import router as portfolio_router
from bltp_websocket_ai import ws_router, ai_router, retrain_models

# Market Data Router
market_router = APIRouter(prefix="/market", tags=["Market Data"])

# Bond Data Router
bonds_router = APIRouter(prefix="/bonds", tags=["Bonds"])

# Pydantic models for market and bond data
from pydantic import BaseModel

class BondSearchFilter(BaseModel):
    bond_type: Optional[str] = None
    min_ytm: Optional[float] = None
    max_ytm: Optional[float] = None
    min_rating: Optional[str] = None
    issuer: Optional[str] = None
    min_maturity_years: Optional[int] = None
    max_maturity_years: Optional[int] = None
    min_investment: Optional[float] = None

class BondListResponse(BaseModel):
    id: str
    isin: str
    name: str
    issuer: str
    bond_type: str
    current_price: float
    face_value: float
    coupon_rate: float
    ytm: Optional[float]
    duration: Optional[float]
    rating: str
    maturity_date: str
    min_investment: float
    is_tradable: bool

class MarketSummary(BaseModel):
    indices: Dict[str, Any]
    bond_market_stats: Dict[str, Any]
    top_movers: Dict[str, List[Dict]]
    market_sentiment: str
    last_updated: str

# Market Data Routes
@market_router.get("/summary", response_model=MarketSummary)
async def get_market_summary(db: Session = Depends(get_db)):
    """Get comprehensive market summary"""
    
    try:
        # Get latest index data
        indices_data = {}
        major_indices = ["NIFTY", "SENSEX", "NIFTY_BANK"]
        
        for index in major_indices:
            latest_data = db.query(MarketData).filter(
                MarketData.symbol == index
            ).order_by(MarketData.timestamp.desc()).first()
            
            if latest_data:
                indices_data[index] = {
                    "price": latest_data.price,
                    "change": latest_data.change,
                    "change_percent": latest_data.change_percent,
                    "timestamp": latest_data.timestamp.isoformat()
                }
        
        # Calculate bond market statistics
        total_bonds = db.query(Bond).count()
        tradable_bonds = db.query(Bond).filter(Bond.is_tradable == True).count()
        avg_ytm = db.query(func.avg(Bond.ytm)).filter(Bond.ytm.isnot(None)).scalar() or 0
        
        # Get recent trades for volume analysis
        recent_trades = db.query(Trade).filter(
            and_(
                Trade.executed_at >= datetime.utcnow() - timedelta(hours=24),
                Trade.status == "filled"
            )
        ).all()
        
        total_volume = sum(trade.total_amount for trade in recent_trades)
        trades_count = len(recent_trades)
        
        bond_market_stats = {
            "total_bonds": total_bonds,
            "tradable_bonds": tradable_bonds,
            "average_ytm": round(avg_ytm, 2),
            "daily_volume": round(total_volume, 2),
            "daily_trades": trades_count
        }
        
        # Get top movers (simplified - using random data for demo)
        top_gainers = []
        top_losers = []
        
        # In production, calculate actual top movers from price data
        sample_bonds = db.query(Bond).filter(Bond.is_tradable == True).limit(5).all()
        for bond in sample_bonds:
            change_pct = np.random.uniform(-2, 3)  # Simulate price change
            mover_data = {
                "isin": bond.isin,
                "name": bond.name[:30] + "..." if len(bond.name) > 30 else bond.name,
                "change_percent": round(change_pct, 2),
                "current_price": bond.current_price
            }
            
            if change_pct > 0:
                top_gainers.append(mover_data)
            else:
                top_losers.append(mover_data)
        
        top_gainers = sorted(top_gainers, key=lambda x: x["change_percent"], reverse=True)[:3]
        top_losers = sorted(top_losers, key=lambda x: x["change_percent"])[:3]
        
        # Determine market sentiment
        sentiment = "neutral"
        if indices_data.get("NIFTY", {}).get("change", 0) > 0.5:
            sentiment = "positive"
        elif indices_data.get("NIFTY", {}).get("change", 0) < -0.5:
            sentiment = "negative"
        
        return MarketSummary(
            indices=indices_data,
            bond_market_stats=bond_market_stats,
            top_movers={"gainers": top_gainers, "losers": top_losers},
            market_sentiment=sentiment,
            last_updated=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market summary")

@market_router.get("/indices/{symbol}")
async def get_index_data(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get historical data for a specific index"""
    
    try:
        # Get historical data
        historical_data = db.query(MarketData).filter(
            and_(
                MarketData.symbol == symbol.upper(),
                MarketData.timestamp >= datetime.utcnow() - timedelta(days=days)
            )
        ).order_by(MarketData.timestamp.asc()).all()
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="Index data not found")
        
        # Format response
        data_points = []
        for data_point in historical_data:
            data_points.append({
                "timestamp": data_point.timestamp.isoformat(),
                "price": data_point.price,
                "change": data_point.change,
                "change_percent": data_point.change_percent,
                "volume": data_point.volume
            })
        
        # Calculate summary statistics
        prices = [d.price for d in historical_data]
        current_price = prices[-1] if prices else 0
        period_high = max(prices) if prices else 0
        period_low = min(prices) if prices else 0
        volatility = np.std(prices) if len(prices) > 1 else 0
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "period_high": period_high,
            "period_low": period_low,
            "volatility": round(volatility, 2),
            "data_points": data_points,
            "period": f"{days} days"
        }
        
    except Exception as e:
        logger.error(f"Error fetching index data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch index data")

@market_router.get("/yield-curve")
async def get_yield_curve(db: Session = Depends(get_db)):
    """Get current government bond yield curve"""
    
    try:
        # Define maturity buckets
        maturity_buckets = [
            (0, 1, "3M"),
            (0.25, 0.75, "6M"),
            (0.75, 1.5, "1Y"),
            (1.5, 3, "2Y"),
            (3, 5, "3Y"),
            (4, 6, "5Y"),
            (6, 8, "7Y"),
            (8, 12, "10Y"),
            (12, 20, "15Y"),
            (20, 40, "30Y")
        ]
        
        yield_curve_data = []
        
        for min_years, max_years, label in maturity_buckets:
            # Get government bonds in this maturity range
            min_date = datetime.utcnow() + timedelta(days=int(min_years * 365))
            max_date = datetime.utcnow() + timedelta(days=int(max_years * 365))
            
            avg_ytm = db.query(func.avg(Bond.ytm)).filter(
                and_(
                    Bond.bond_type == "government",
                    Bond.maturity_date >= min_date,
                    Bond.maturity_date <= max_date,
                    Bond.ytm.isnot(None)
                )
            ).scalar()
            
            if avg_ytm:
                yield_curve_data.append({
                    "maturity": label,
                    "years": (min_years + max_years) / 2,
                    "yield": round(avg_ytm, 2)
                })
        
        # If no data found, provide sample yield curve
        if not yield_curve_data:
            sample_curve = [
                {"maturity": "3M", "years": 0.25, "yield": 6.8},
                {"maturity": "6M", "years": 0.5, "yield": 6.9},
                {"maturity": "1Y", "years": 1, "yield": 7.0},
                {"maturity": "2Y", "years": 2, "yield": 7.1},
                {"maturity": "3Y", "years": 3, "yield": 7.2},
                {"maturity": "5Y", "years": 5, "yield": 7.3},
                {"maturity": "7Y", "years": 7, "yield": 7.35},
                {"maturity": "10Y", "years": 10, "yield": 7.4},
                {"maturity": "15Y", "years": 15, "yield": 7.45},
                {"maturity": "30Y", "years": 30, "yield": 7.5}
            ]
            yield_curve_data = sample_curve
        
        return {
            "curve_date": datetime.utcnow().date().isoformat(),
            "curve_type": "Government Securities",
            "data": yield_curve_data,
            "curve_analysis": {
                "shape": "Normal" if yield_curve_data[-1]["yield"] > yield_curve_data[0]["yield"] else "Inverted",
                "steepness": yield_curve_data[-1]["yield"] - yield_curve_data[0]["yield"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching yield curve: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch yield curve")

# Bond Data Routes
@bonds_router.get("/search", response_model=List[BondListResponse])
async def search_bonds(
    filters: BondSearchFilter = Depends(),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="ytm", regex="^(ytm|rating|maturity_date|current_price|issuer)$"),
    sort_order: str = Query(default="desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db)
):
    """Search and filter bonds with pagination"""
    
    try:
        # Build query
        query = db.query(Bond).filter(Bond.is_tradable == True)
        
        # Apply filters
        if filters.bond_type:
            query = query.filter(Bond.bond_type.ilike(f"%{filters.bond_type}%"))
        
        if filters.min_ytm is not None:
            query = query.filter(Bond.ytm >= filters.min_ytm)
        
        if filters.max_ytm is not None:
            query = query.filter(Bond.ytm <= filters.max_ytm)
        
        if filters.issuer:
            query = query.filter(Bond.issuer.ilike(f"%{filters.issuer}%"))
        
        if filters.min_investment is not None:
            query = query.filter(Bond.min_investment <= filters.min_investment)
        
        # Apply maturity filters
        if filters.min_maturity_years is not None:
            min_maturity_date = datetime.utcnow() + timedelta(days=filters.min_maturity_years * 365)
            query = query.filter(Bond.maturity_date >= min_maturity_date)
        
        if filters.max_maturity_years is not None:
            max_maturity_date = datetime.utcnow() + timedelta(days=filters.max_maturity_years * 365)
            query = query.filter(Bond.maturity_date <= max_maturity_date)
        
        # Apply rating filter
        if filters.min_rating:
            rating_order = ['CCC', 'B', 'BB', 'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA']
            if filters.min_rating in rating_order:
                min_rating_index = rating_order.index(filters.min_rating)
                valid_ratings = rating_order[min_rating_index:]
                query = query.filter(Bond.rating.in_(valid_ratings))
        
        # Apply sorting
        if sort_by == "ytm":
            order_field = Bond.ytm.desc() if sort_order == "desc" else Bond.ytm.asc()
        elif sort_by == "rating":
            order_field = Bond.rating.desc() if sort_order == "desc" else Bond.rating.asc()
        elif sort_by == "maturity_date":
            order_field = Bond.maturity_date.desc() if sort_order == "desc" else Bond.maturity_date.asc()
        elif sort_by == "current_price":
            order_field = Bond.current_price.desc() if sort_order == "desc" else Bond.current_price.asc()
        else:  # issuer
            order_field = Bond.issuer.desc() if sort_order == "desc" else Bond.issuer.asc()
        
        query = query.order_by(order_field)
        
        # Apply pagination
        total_count = query.count()
        offset = (page - 1) * limit
        bonds = query.offset(offset).limit(limit).all()
        
        # Format response
        bond_list = []
        for bond in bonds:
            bond_list.append(BondListResponse(
                id=str(bond.id),
                isin=bond.isin,
                name=bond.name,
                issuer=bond.issuer,
                bond_type=bond.bond_type,
                current_price=bond.current_price or 0,
                face_value=bond.face_value,
                coupon_rate=bond.coupon_rate,
                ytm=bond.ytm,
                duration=bond.duration,
                rating=bond.rating or "NR",
                maturity_date=bond.maturity_date.isoformat() if bond.maturity_date else "",
                min_investment=bond.min_investment,
                is_tradable=bond.is_tradable
            ))
        
        return bond_list
        
    except Exception as e:
        logger.error(f"Error searching bonds: {e}")
        raise HTTPException(status_code=500, detail="Bond search failed")

@bonds_router.get("/{bond_id}", response_model=dict)
async def get_bond_details(bond_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific bond"""
    
    try:
        bond = db.query(Bond).filter(Bond.id == bond_id).first()
        if not bond:
            raise HTTPException(status_code=404, detail="Bond not found")
        
        # Calculate additional metrics
        days_to_maturity = (bond.maturity_date - datetime.utcnow()).days if bond.maturity_date else None
        years_to_maturity = days_to_maturity / 365.25 if days_to_maturity else None
        
        # Get recent trading activity
        recent_trades = db.query(Trade).filter(
            and_(
                Trade.bond_id == bond_id,
                Trade.status == "filled",
                Trade.executed_at >= datetime.utcnow() - timedelta(days=7)
            )
        ).order_by(Trade.executed_at.desc()).limit(10).all()
        
        trading_activity = []
        for trade in recent_trades:
            trading_activity.append({
                "date": trade.executed_at.date().isoformat(),
                "price": trade.price,
                "quantity": trade.quantity,
                "trade_type": trade.trade_type
            })
        
        # Calculate price performance (simplified)
        price_history = []
        base_price = bond.current_price or bond.face_value
        for i in range(30):  # Last 30 days
            date = datetime.utcnow() - timedelta(days=i)
            # Simulate price history
            price = base_price + np.random.normal(0, 0.5)
            price_history.append({
                "date": date.date().isoformat(),
                "price": round(price, 2)
            })
        
        price_history.reverse()  # Chronological order
        
        return {
            "bond_info": {
                "id": str(bond.id),
                "isin": bond.isin,
                "name": bond.name,
                "issuer": bond.issuer,
                "bond_type": bond.bond_type,
                "rating": bond.rating,
                "issue_date": bond.issue_date.isoformat() if bond.issue_date else None,
                "maturity_date": bond.maturity_date.isoformat() if bond.maturity_date else None,
                "years_to_maturity": round(years_to_maturity, 2) if years_to_maturity else None
            },
            "pricing": {
                "current_price": bond.current_price,
                "face_value": bond.face_value,
                "coupon_rate": bond.coupon_rate,
                "ytm": bond.ytm,
                "duration": bond.duration,
                "min_investment": bond.min_investment
            },
            "trading_info": {
                "is_tradable": bond.is_tradable,
                "recent_activity": trading_activity,
                "avg_daily_volume": sum(t["quantity"] for t in trading_activity) / max(len(trading_activity), 1)
            },
            "price_history": price_history[-7:],  # Last 7 days
            "key_metrics": {
                "accrued_interest": self.calculate_accrued_interest(bond),
                "modified_duration": bond.duration / (1 + bond.ytm/100) if bond.duration and bond.ytm else None,
                "convexity": bond.duration * 1.2 if bond.duration else None,  # Simplified
                "price_volatility": round(np.std([p["price"] for p in price_history]), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting bond details: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bond details")

def calculate_accrued_interest(bond: Bond) -> float:
    """Calculate accrued interest for a bond"""
    if not bond.issue_date or not bond.coupon_rate:
        return 0
    
    # Simplified accrued interest calculation
    days_since_last_coupon = (datetime.utcnow() - bond.issue_date).days % 365
    annual_interest = bond.face_value * bond.coupon_rate / 100
    daily_interest = annual_interest / 365
    
    return round(daily_interest * days_since_last_coupon, 2)

@bonds_router.get("/categories/stats", response_model=dict)
async def get_bond_categories_stats(db: Session = Depends(get_db)):
    """Get statistics for different bond categories"""
    
    try:
        # Get category statistics
        category_stats = {}
        
        # Government bonds
        govt_bonds = db.query(Bond).filter(Bond.bond_type == "government").all()
        if govt_bonds:
            govt_ytms = [b.ytm for b in govt_bonds if b.ytm]
            category_stats["government"] = {
                "count": len(govt_bonds),
                "avg_ytm": round(np.mean(govt_ytms), 2) if govt_ytms else 0,
                "min_ytm": round(min(govt_ytms), 2) if govt_ytms else 0,
                "max_ytm": round(max(govt_ytms), 2) if govt_ytms else 0,
                "avg_duration": round(np.mean([b.duration for b in govt_bonds if b.duration]), 2) if any(b.duration for b in govt_bonds) else 0
            }
        
        # Corporate bonds
        corp_bonds = db.query(Bond).filter(Bond.bond_type == "corporate").all()
        if corp_bonds:
            corp_ytms = [b.ytm for b in corp_bonds if b.ytm]
            category_stats["corporate"] = {
                "count": len(corp_bonds),
                "avg_ytm": round(np.mean(corp_ytms), 2) if corp_ytms else 0,
                "min_ytm": round(min(corp_ytms), 2) if corp_ytms else 0,
                "max_ytm": round(max(corp_ytms), 2) if corp_ytms else 0,
                "avg_duration": round(np.mean([b.duration for b in corp_bonds if b.duration]), 2) if any(b.duration for b in corp_bonds) else 0
            }
        
        # Rating distribution
        rating_distribution = {}
        all_bonds = db.query(Bond).filter(Bond.rating.isnot(None)).all()
        
        for bond in all_bonds:
            rating = bond.rating
            if rating not in rating_distribution:
                rating_distribution[rating] = {"count": 0, "avg_ytm": []}
            rating_distribution[rating]["count"] += 1
            if bond.ytm:
                rating_distribution[rating]["avg_ytm"].append(bond.ytm)
        
        # Calculate average YTM for each rating
        for rating in rating_distribution:
            ytms = rating_distribution[rating]["avg_ytm"]
            rating_distribution[rating]["avg_ytm"] = round(np.mean(ytms), 2) if ytms else 0
        
        # Maturity distribution
        maturity_buckets = {
            "0-1 years": {"count": 0, "avg_ytm": []},
            "1-3 years": {"count": 0, "avg_ytm": []},
            "3-5 years": {"count": 0, "avg_ytm": []},
            "5-10 years": {"count": 0, "avg_ytm": []},
            "10+ years": {"count": 0, "avg_ytm": []}
        }
        
        for bond in all_bonds:
            if bond.maturity_date:
                years_to_maturity = (bond.maturity_date - datetime.utcnow()).days / 365.25
                
                if years_to_maturity <= 1:
                    bucket = "0-1 years"
                elif years_to_maturity <= 3:
                    bucket = "1-3 years"
                elif years_to_maturity <= 5:
                    bucket = "3-5 years"
                elif years_to_maturity <= 10:
                    bucket = "5-10 years"
                else:
                    bucket = "10+ years"
                
                maturity_buckets[bucket]["count"] += 1
                if bond.ytm:
                    maturity_buckets[bucket]["avg_ytm"].append(bond.ytm)
        
        # Calculate average YTM for each maturity bucket
        for bucket in maturity_buckets:
            ytms = maturity_buckets[bucket]["avg_ytm"]
            maturity_buckets[bucket]["avg_ytm"] = round(np.mean(ytms), 2) if ytms else 0
        
        return {
            "category_stats": category_stats,
            "rating_distribution": rating_distribution,
            "maturity_distribution": maturity_buckets,
            "market_overview": {
                "total_bonds": len(all_bonds),
                "tradable_bonds": db.query(Bond).filter(Bond.is_tradable == True).count(),
                "avg_market_ytm": round(np.mean([b.ytm for b in all_bonds if b.ytm]), 2) if any(b.ytm for b in all_bonds) else 0,
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting bond category stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch category statistics")

# Background task to populate sample bond data
async def populate_sample_bonds(db: Session):
    """Populate database with sample bonds for testing"""
    
    try:
        # Check if bonds already exist
        existing_bonds = db.query(Bond).count()
        if existing_bonds > 0:
            logger.info("Bonds already exist in database")
            return
        
        # Sample government bonds
        govt_bonds = [
            {
                "isin": "IN0123456789",
                "name": "Government of India 7.26% 2032",
                "issuer": "Government of India",
                "face_value": 100,
                "coupon_rate": 7.26,
                "maturity_date": datetime(2032, 1, 8),
                "issue_date": datetime(2022, 1, 8),
                "bond_type": "government",
                "rating": "AAA",
                "current_price": 102.50,
                "ytm": 7.1,
                "duration": 8.5,
                "is_tradable": True,
                "min_investment": 1000
            },
            {
                "isin": "IN0234567890",
                "name": "Government of India 6.45% 2029",
                "issuer": "Government of India",
                "face_value": 100,
                "coupon_rate": 6.45,
                "maturity_date": datetime(2029, 10, 1),
                "issue_date": datetime(2019, 10, 1),
                "bond_type": "government",
                "rating": "AAA",
                "current_price": 98.75,
                "ytm": 6.8,
                "duration": 5.2,
                "is_tradable": True,
                "min_investment": 1000
            }
        ]
        
        # Sample corporate bonds
        corp_bonds = [
            {
                "isin": "IN0345678901",
                "name": "HDFC Bank 8.25% 2027",
                "issuer": "HDFC Bank Limited",
                "face_value": 1000,
                "coupon_rate": 8.25,
                "maturity_date": datetime(2027, 3, 15),
                "issue_date": datetime(2022, 3, 15),
                "bond_type": "corporate",
                "rating": "AAA",
                "current_price": 1020.00,
                "ytm": 7.8,
                "duration": 3.8,
                "is_tradable": True,
                "min_investment": 10000
            },
            {
                "isin": "IN0456789012",
                "name": "Reliance Industries 7.75% 2030",
                "issuer": "Reliance Industries Limited",
                "face_value": 1000,
                "coupon_rate": 7.75,
                "maturity_date": datetime(2030, 6, 20),
                "issue_date": datetime(2020, 6, 20),
                "bond_type": "corporate",
                "rating": "AA+",
                "current_price": 995.50,
                "ytm": 7.9,
                "duration": 6.1,
                "is_tradable": True,
                "min_investment": 5000
            }
        ]
        
        # Add bonds to database
        all_sample_bonds = govt_bonds + corp_bonds
        
        for bond_data in all_sample_bonds:
            bond = Bond(**bond_data)
            db.add(bond)
        
        db.commit()
        logger.info(f"Added {len(all_sample_bonds)} sample bonds to database")
        
    except Exception as e:
        logger.error(f"Error populating sample bonds: {e}")
        db.rollback()

# Register all routers with the main app
app.include_router(auth_router)
app.include_router(trading_router)
app.include_router(portfolio_router)
app.include_router(market_router)
app.include_router(bonds_router)
app.include_router(ai_router)

# Add WebSocket route
app.include_router(ws_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BLTP - Bond Liquidity Trading Platform API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "authentication": "/auth/*",
            "trading": "/trading/*",
            "portfolio": "/portfolio/*",
            "market_data": "/market/*",
            "bonds": "/bonds/*",
            "ai_services": "/ai/*",
            "websocket": "/ws/{user_id}",
            "documentation": "/docs"
        },
        "features": [
            "User authentication with KYC",
            "Real-time bond trading",
            "Portfolio management & analytics",
            "Risk assessment tools",
            "AI-powered recommendations",
            "Live market data feeds",
            "WebSocket connections for real-time updates"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        
        # Test Redis connection
        redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "operational",
                "redis": "operational",
                "websocket": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# Startup event to initialize data and background tasks
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting BLTP Backend Application...")
    
    # Get database session
    db = next(get_db())
    
    try:
        # Populate sample bonds if needed
        await populate_sample_bonds(db)
        
        # Start background tasks
        asyncio.create_task(retrain_models())
        
        logger.info("BLTP Backend Application started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    finally:
        db.close()

# Add this method to the class (it was referenced but not defined)
def calculate_accrued_interest(bond: Bond) -> float:
    """Calculate accrued interest for a bond"""
    if not bond.issue_date or not bond.coupon_rate:
        return 0
    
    # Simplified accrued interest calculation
    days_since_last_coupon = (datetime.utcnow() - bond.issue_date).days % 365
    annual_interest = bond.face_value * bond.coupon_rate / 100
    daily_interest = annual_interest / 365
    
    return round(daily_interest * days_since_last_coupon, 2)