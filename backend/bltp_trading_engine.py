"""
BLTP Platform Backend - Bond Trading Engine
Handles bond trading, order matching, portfolio management
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
from decimal import Decimal, ROUND_HALF_UP
import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from dataclasses import dataclass

# Import from main app
from main import (
    get_db, get_current_user, User, Bond, Trade, Portfolio, Holding, 
    TradeRequest, redis_client, manager, logger, MarketData
)

router = APIRouter(prefix="/trading", tags=["Trading"])

# Trading Engine Classes and Enums
class OrderStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderBook:
    bond_id: str
    buy_orders: List[Dict] = None
    sell_orders: List[Dict] = None
    
    def __post_init__(self):
        if self.buy_orders is None:
            self.buy_orders = []
        if self.sell_orders is None:
            self.sell_orders = []

# Trading Engine Core
class BondTradingEngine:
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.pending_orders: Dict[str, Dict] = {}
        
    async def place_order(self, order_data: Dict, db: Session) -> Dict:
        """Place a new trading order"""
        try:
            bond_id = order_data["bond_id"]
            user_id = order_data["user_id"]
            trade_type = order_data["trade_type"]
            quantity = Decimal(str(order_data["quantity"]))
            order_type = order_data.get("order_type", "market")
            limit_price = order_data.get("price")
            
            # Get bond details
            bond = db.query(Bond).filter(Bond.id == bond_id).first()
            if not bond:
                raise ValueError("Bond not found")
            
            if not bond.is_tradable:
                raise ValueError("Bond is not tradable")
            
            # Validate minimum investment
            total_value = quantity * (limit_price or bond.current_price)
            if total_value < bond.min_investment:
                raise ValueError(f"Minimum investment amount is ₹{bond.min_investment}")
            
            # Check user's portfolio and cash balance for buy orders
            user_portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
            if not user_portfolio:
                # Create default portfolio
                user_portfolio = Portfolio(user_id=user_id, name="Default Portfolio")
                db.add(user_portfolio)
                db.commit()
                db.refresh(user_portfolio)
            
            if trade_type == TradeType.BUY.value:
                required_cash = total_value
                if user_portfolio.cash_balance < required_cash:
                    raise ValueError("Insufficient cash balance")
            else:  # SELL
                # Check if user has enough bonds to sell
                holding = db.query(Holding).filter(
                    and_(
                        Holding.portfolio_id == user_portfolio.id,
                        Holding.bond_id == bond_id
                    )
                ).first()
                
                if not holding or holding.quantity < quantity:
                    raise ValueError("Insufficient bond holdings")
            
            # Create trade record
            trade = Trade(
                user_id=user_id,
                bond_id=bond_id,
                trade_type=trade_type,
                quantity=float(quantity),
                price=limit_price or bond.current_price,
                total_amount=float(total_value),
                order_type=order_type,
                status=OrderStatus.PENDING.value
            )
            
            db.add(trade)
            db.commit()
            db.refresh(trade)
            
            # Add to order book
            if bond_id not in self.order_books:
                self.order_books[bond_id] = OrderBook(bond_id=bond_id)
            
            order_entry = {
                "trade_id": str(trade.id),
                "user_id": user_id,
                "quantity": float(quantity),
                "price": limit_price or bond.current_price,
                "timestamp": datetime.utcnow(),
                "order_type": order_type
            }
            
            if trade_type == TradeType.BUY.value:
                self.order_books[bond_id].buy_orders.append(order_entry)
                # Sort buy orders by price (descending) and time (ascending)
                self.order_books[bond_id].buy_orders.sort(
                    key=lambda x: (-x["price"], x["timestamp"])
                )
            else:
                self.order_books[bond_id].sell_orders.append(order_entry)
                # Sort sell orders by price (ascending) and time (ascending)
                self.order_books[bond_id].sell_orders.sort(
                    key=lambda x: (x["price"], x["timestamp"])
                )
            
            # Store pending order
            self.pending_orders[str(trade.id)] = order_entry
            
            # Try to match orders
            await self.match_orders(bond_id, db)
            
            return {
                "trade_id": str(trade.id),
                "status": trade.status,
                "message": "Order placed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise ValueError(str(e))
    
    async def match_orders(self, bond_id: str, db: Session):
        """Match buy and sell orders"""
        try:
            if bond_id not in self.order_books:
                return
            
            order_book = self.order_books[bond_id]
            buy_orders = order_book.buy_orders.copy()
            sell_orders = order_book.sell_orders.copy()
            
            matches = []
            
            for buy_order in buy_orders:
                for sell_order in sell_orders:
                    # Check if orders can be matched
                    if buy_order["price"] >= sell_order["price"]:
                        # Determine execution price (typically the limit order price)
                        execution_price = sell_order["price"]
                        
                        # Determine quantity to trade
                        trade_quantity = min(buy_order["quantity"], sell_order["quantity"])
                        
                        # Create match
                        match = {
                            "buy_trade_id": buy_order["trade_id"],
                            "sell_trade_id": sell_order["trade_id"],
                            "buy_user_id": buy_order["user_id"],
                            "sell_user_id": sell_order["user_id"],
                            "quantity": trade_quantity,
                            "price": execution_price,
                            "total_amount": trade_quantity * execution_price
                        }
                        
                        matches.append(match)
                        
                        # Update order quantities
                        buy_order["quantity"] -= trade_quantity
                        sell_order["quantity"] -= trade_quantity
                        
                        # Remove fully filled orders
                        if buy_order["quantity"] == 0:
                            break
                        
                        if sell_order["quantity"] == 0:
                            continue
            
            # Execute matches
            for match in matches:
                await self.execute_trade(match, bond_id, db)
            
            # Update order book
            order_book.buy_orders = [o for o in buy_orders if o["quantity"] > 0]
            order_book.sell_orders = [o for o in sell_orders if o["quantity"] > 0]
            
        except Exception as e:
            logger.error(f"Error matching orders: {e}")
    
    async def execute_trade(self, match: Dict, bond_id: str, db: Session):
        """Execute a matched trade"""
        try:
            buy_trade = db.query(Trade).filter(Trade.id == match["buy_trade_id"]).first()
            sell_trade = db.query(Trade).filter(Trade.id == match["sell_trade_id"]).first()
            
            if not buy_trade or not sell_trade:
                return
            
            # Update trade records
            execution_time = datetime.utcnow()
            
            buy_trade.executed_at = execution_time
            buy_trade.price = match["price"]
            buy_trade.total_amount = match["total_amount"]
            
            sell_trade.executed_at = execution_time
            sell_trade.price = match["price"]
            sell_trade.total_amount = match["total_amount"]
            
            # Check if trades are fully filled
            if match["quantity"] == buy_trade.quantity:
                buy_trade.status = OrderStatus.FILLED.value
            else:
                buy_trade.status = OrderStatus.PARTIALLY_FILLED.value
            
            if match["quantity"] == sell_trade.quantity:
                sell_trade.status = OrderStatus.FILLED.value
            else:
                sell_trade.status = OrderStatus.PARTIALLY_FILLED.value
            
            # Update portfolios
            await self.update_portfolios(match, bond_id, db)
            
            # Update market price
            bond = db.query(Bond).filter(Bond.id == bond_id).first()
            if bond:
                bond.current_price = match["price"]
            
            db.commit()
            
            # Broadcast trade execution
            await manager.broadcast(json.dumps({
                "type": "trade_executed",
                "data": {
                    "bond_id": bond_id,
                    "price": match["price"],
                    "quantity": match["quantity"],
                    "timestamp": execution_time.isoformat()
                }
            }))
            
            logger.info(f"Trade executed: {match}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            db.rollback()
    
    async def update_portfolios(self, match: Dict, bond_id: str, db: Session):
        """Update user portfolios after trade execution"""
        try:
            # Get buyer and seller portfolios
            buyer_portfolio = db.query(Portfolio).filter(Portfolio.user_id == match["buy_user_id"]).first()
            seller_portfolio = db.query(Portfolio).filter(Portfolio.user_id == match["sell_user_id"]).first()
            
            trade_amount = match["total_amount"]
            quantity = match["quantity"]
            price = match["price"]
            
            # Update buyer portfolio
            if buyer_portfolio:
                # Deduct cash
                buyer_portfolio.cash_balance -= trade_amount
                
                # Add or update bond holding
                buyer_holding = db.query(Holding).filter(
                    and_(
                        Holding.portfolio_id == buyer_portfolio.id,
                        Holding.bond_id == bond_id
                    )
                ).first()
                
                if buyer_holding:
                    # Update existing holding
                    total_quantity = buyer_holding.quantity + quantity
                    total_cost = (buyer_holding.quantity * buyer_holding.avg_price) + trade_amount
                    buyer_holding.avg_price = total_cost / total_quantity
                    buyer_holding.quantity = total_quantity
                else:
                    # Create new holding
                    buyer_holding = Holding(
                        portfolio_id=buyer_portfolio.id,
                        bond_id=bond_id,
                        quantity=quantity,
                        avg_price=price,
                        current_value=trade_amount
                    )
                    db.add(buyer_holding)
            
            # Update seller portfolio
            if seller_portfolio:
                # Add cash
                seller_portfolio.cash_balance += trade_amount
                
                # Update bond holding
                seller_holding = db.query(Holding).filter(
                    and_(
                        Holding.portfolio_id == seller_portfolio.id,
                        Holding.bond_id == bond_id
                    )
                ).first()
                
                if seller_holding:
                    seller_holding.quantity -= quantity
                    
                    # Calculate realized P&L
                    cost_basis = seller_holding.avg_price * quantity
                    realized_pnl = trade_amount - cost_basis
                    seller_portfolio.realized_pnl += realized_pnl
                    
                    # Remove holding if quantity becomes zero
                    if seller_holding.quantity <= 0:
                        db.delete(seller_holding)
            
            # Update portfolio values
            await self.calculate_portfolio_values(buyer_portfolio.id, db)
            await self.calculate_portfolio_values(seller_portfolio.id, db)
            
        except Exception as e:
            logger.error(f"Error updating portfolios: {e}")
    
    async def calculate_portfolio_values(self, portfolio_id: str, db: Session):
        """Calculate portfolio total value and unrealized P&L"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio:
                return
            
            holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
            
            total_value = portfolio.cash_balance
            unrealized_pnl = 0
            
            for holding in holdings:
                bond = db.query(Bond).filter(Bond.id == holding.bond_id).first()
                if bond:
                    current_value = holding.quantity * bond.current_price
                    cost_basis = holding.quantity * holding.avg_price
                    
                    holding.current_value = current_value
                    holding.unrealized_pnl = current_value - cost_basis
                    
                    total_value += current_value
                    unrealized_pnl += holding.unrealized_pnl
            
            portfolio.total_value = total_value
            portfolio.unrealized_pnl = unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating portfolio values: {e}")
    
    async def cancel_order(self, trade_id: str, user_id: str, db: Session) -> bool:
        """Cancel a pending order"""
        try:
            trade = db.query(Trade).filter(
                and_(
                    Trade.id == trade_id,
                    Trade.user_id == user_id,
                    Trade.status == OrderStatus.PENDING.value
                )
            ).first()
            
            if not trade:
                return False
            
            # Update trade status
            trade.status = OrderStatus.CANCELLED.value
            
            # Remove from order book
            bond_id = str(trade.bond_id)
            if bond_id in self.order_books:
                order_book = self.order_books[bond_id]
                
                if trade.trade_type == TradeType.BUY.value:
                    order_book.buy_orders = [
                        o for o in order_book.buy_orders 
                        if o["trade_id"] != trade_id
                    ]
                else:
                    order_book.sell_orders = [
                        o for o in order_book.sell_orders 
                        if o["trade_id"] != trade_id
                    ]
            
            # Remove from pending orders
            if trade_id in self.pending_orders:
                del self.pending_orders[trade_id]
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

# Initialize trading engine
trading_engine = BondTradingEngine()

# Pydantic models for API
from pydantic import BaseModel, validator

class OrderRequest(BaseModel):
    bond_id: str
    trade_type: str
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None
    
    @validator('trade_type')
    def validate_trade_type(cls, v):
        if v not in [TradeType.BUY.value, TradeType.SELL.value]:
            raise ValueError('Invalid trade type')
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        valid_types = [OrderType.MARKET.value, OrderType.LIMIT.value]
        if v not in valid_types:
            raise ValueError('Invalid order type')
        return v
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v

class TradeResponse(BaseModel):
    id: str
    bond_id: str
    trade_type: str
    quantity: float
    price: float
    total_amount: float
    status: str
    order_type: str
    created_at: str
    executed_at: Optional[str]

class OrderBookResponse(BaseModel):
    bond_id: str
    buy_orders: List[Dict]
    sell_orders: List[Dict]
    last_traded_price: float
    best_bid: Optional[float]
    best_ask: Optional[float]

# Trading API Routes
@router.post("/place-order", response_model=dict)
async def place_order(
    order: OrderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Place a new trading order"""
    
    if current_user.kyc_status != "verified":
        raise HTTPException(
            status_code=400,
            detail="KYC verification required for trading"
        )
    
    try:
        order_data = {
            "bond_id": order.bond_id,
            "user_id": str(current_user.id),
            "trade_type": order.trade_type,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "price": order.price
        }
        
        result = await trading_engine.place_order(order_data, db)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/orders", response_model=List[TradeResponse])
async def get_user_orders(
    status: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's trading orders"""
    
    query = db.query(Trade).filter(Trade.user_id == current_user.id)
    
    if status:
        query = query.filter(Trade.status == status)
    
    trades = query.order_by(Trade.created_at.desc()).limit(limit).all()
    
    return [
        TradeResponse(
            id=str(trade.id),
            bond_id=str(trade.bond_id),
            trade_type=trade.trade_type,
            quantity=trade.quantity,
            price=trade.price,
            total_amount=trade.total_amount,
            status=trade.status,
            order_type=trade.order_type,
            created_at=trade.created_at.isoformat(),
            executed_at=trade.executed_at.isoformat() if trade.executed_at else None
        )
        for trade in trades
    ]

@router.delete("/orders/{trade_id}", response_model=dict)
async def cancel_order(
    trade_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel a pending order"""
    
    success = await trading_engine.cancel_order(trade_id, str(current_user.id), db)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Order not found or cannot be cancelled"
        )
    
    return {"message": "Order cancelled successfully"}

@router.get("/order-book/{bond_id}", response_model=OrderBookResponse)
async def get_order_book(
    bond_id: str,
    db: Session = Depends(get_db)
):
    """Get order book for a specific bond"""
    
    bond = db.query(Bond).filter(Bond.id == bond_id).first()
    if not bond:
        raise HTTPException(status_code=404, detail="Bond not found")
    
    order_book = trading_engine.order_books.get(bond_id, OrderBook(bond_id=bond_id))
    
    # Get best bid and ask
    best_bid = max([o["price"] for o in order_book.buy_orders], default=None)
    best_ask = min([o["price"] for o in order_book.sell_orders], default=None)
    
    return OrderBookResponse(
        bond_id=bond_id,
        buy_orders=order_book.buy_orders[:10],  # Top 10 bids
        sell_orders=order_book.sell_orders[:10],  # Top 10 asks
        last_traded_price=bond.current_price,
        best_bid=best_bid,
        best_ask=best_ask
    )