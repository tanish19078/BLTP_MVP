"""
BLTP Platform Backend - Portfolio Management & Analytics Routes
Handles portfolio operations, risk analysis, and performance tracking
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats

# Import from main app
from main import (
    get_db, get_current_user, User, Portfolio, Holding, Bond, Trade,
    redis_client, logger
)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

# Pydantic models for portfolio
from pydantic import BaseModel

class PortfolioSummary(BaseModel):
    id: str
    name: str
    total_value: float
    cash_balance: float
    invested_amount: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    return_percentage: float
    holdings_count: int

class HoldingResponse(BaseModel):
    id: str
    bond_id: str
    bond_name: str
    bond_isin: str
    issuer: str
    quantity: float
    avg_price: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    return_percentage: float
    weight: float  # Portfolio weight

class RiskAnalysis(BaseModel):
    portfolio_duration: float
    modified_duration: float
    convexity: float
    var_1_day: float  # Value at Risk (1 day, 95% confidence)
    var_10_day: float  # Value at Risk (10 day, 95% confidence)
    beta: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float

class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

class SectorAllocation(BaseModel):
    sector: str
    allocation: float
    value: float

class MaturityAllocation(BaseModel):
    maturity_bucket: str
    allocation: float
    value: float

# Portfolio Analytics Engine
class PortfolioAnalytics:
    def __init__(self):
        self.risk_free_rate = 0.06  # 6% risk-free rate (approximate for India)
    
    def calculate_portfolio_duration(self, holdings: List[Holding], bonds: List[Bond]) -> float:
        """Calculate portfolio duration"""
        try:
            total_value = sum(h.current_value for h in holdings)
            if total_value == 0:
                return 0
            
            weighted_duration = 0
            bond_map = {str(b.id): b for b in bonds}
            
            for holding in holdings:
                bond = bond_map.get(str(holding.bond_id))
                if bond and bond.duration:
                    weight = holding.current_value / total_value
                    weighted_duration += weight * bond.duration
            
            return weighted_duration
        except Exception as e:
            logger.error(f"Error calculating duration: {e}")
            return 0
    
    def calculate_modified_duration(self, duration: float, ytm: float) -> float:
        """Calculate modified duration"""
        return duration / (1 + ytm / 100)
    
    def calculate_convexity(self, holdings: List[Holding], bonds: List[Bond]) -> float:
        """Calculate portfolio convexity"""
        # Simplified convexity calculation
        return self.calculate_portfolio_duration(holdings, bonds) * 1.2
    
    def calculate_var(self, portfolio_value: float, volatility: float, days: int = 1, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        z_score = stats.norm.ppf(1 - confidence)
        daily_var = portfolio_value * volatility * z_score / np.sqrt(252)  # Assuming 252 trading days
        return daily_var * np.sqrt(days)
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        
        rf_rate = risk_free_rate or self.risk_free_rate
        excess_returns = [r - rf_rate/252 for r in returns]  # Daily risk-free rate
        
        if len(excess_returns) < 2:
            return 0
        
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return == 0:
            return 0
        
        return (mean_excess_return * 252) / (std_excess_return * np.sqrt(252))  # Annualized
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0
        
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def get_sector_allocation(self, holdings: List[Holding], bonds: List[Bond]) -> List[SectorAllocation]:
        """Calculate sector allocation"""
        sector_map = {}
        total_value = sum(h.current_value for h in holdings)
        bond_map = {str(b.id): b for b in bonds}
        
        for holding in holdings:
            bond = bond_map.get(str(holding.bond_id))
            if bond:
                # Map bond type to sector
                sector = self.get_sector_from_bond_type(bond.bond_type, bond.issuer)
                if sector not in sector_map:
                    sector_map[sector] = 0
                sector_map[sector] += holding.current_value
        
        return [
            SectorAllocation(
                sector=sector,
                value=value,
                allocation=value / total_value * 100 if total_value > 0 else 0
            )
            for sector, value in sector_map.items()
        ]
    
    def get_maturity_allocation(self, holdings: List[Holding], bonds: List[Bond]) -> List[MaturityAllocation]:
        """Calculate maturity allocation"""
        maturity_map = {
            "0-1 years": 0,
            "1-3 years": 0,
            "3-5 years": 0,
            "5-10 years": 0,
            "10+ years": 0
        }
        
        total_value = sum(h.current_value for h in holdings)
        bond_map = {str(b.id): b for b in bonds}
        
        for holding in holdings:
            bond = bond_map.get(str(holding.bond_id))
            if bond and bond.maturity_date:
                years_to_maturity = (bond.maturity_date - datetime.utcnow()).days / 365.25
                bucket = self.get_maturity_bucket(years_to_maturity)
                maturity_map[bucket] += holding.current_value
        
        return [
            MaturityAllocation(
                maturity_bucket=bucket,
                value=value,
                allocation=value / total_value * 100 if total_value > 0 else 0
            )
            for bucket, value in maturity_map.items()
            if value > 0
        ]
    
    def get_sector_from_bond_type(self, bond_type: str, issuer: str) -> str:
        """Map bond type and issuer to sector"""
        if bond_type.lower() == "government":
            return "Government"
        elif bond_type.lower() == "corporate":
            # Simple sector mapping based on issuer name
            issuer_lower = issuer.lower()
            if any(word in issuer_lower for word in ["bank", "financial", "nbfc"]):
                return "BFSI"
            elif any(word in issuer_lower for word in ["infra", "power", "energy"]):
                return "Infrastructure"
            elif any(word in issuer_lower for word in ["it", "tech", "software"]):
                return "IT"
            else:
                return "Others"
        else:
            return "Others"
    
    def get_maturity_bucket(self, years_to_maturity: float) -> str:
        """Get maturity bucket based on years to maturity"""
        if years_to_maturity <= 1:
            return "0-1 years"
        elif years_to_maturity <= 3:
            return "1-3 years"
        elif years_to_maturity <= 5:
            return "3-5 years"
        elif years_to_maturity <= 10:
            return "5-10 years"
        else:
            return "10+ years"

# Initialize analytics engine
analytics = PortfolioAnalytics()

# Portfolio API Routes
@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio summary for the current user"""
    
    # Get or create default portfolio
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        portfolio = Portfolio(
            user_id=current_user.id,
            name="Default Portfolio"
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
    
    # Get holdings
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    
    # Calculate invested amount
    invested_amount = 0
    for holding in holdings:
        invested_amount += holding.quantity * holding.avg_price
    
    # Calculate total return
    total_return = portfolio.realized_pnl + portfolio.unrealized_pnl
    return_percentage = (total_return / invested_amount * 100) if invested_amount > 0 else 0
    
    return PortfolioSummary(
        id=str(portfolio.id),
        name=portfolio.name,
        total_value=portfolio.total_value,
        cash_balance=portfolio.cash_balance,
        invested_amount=invested_amount,
        unrealized_pnl=portfolio.unrealized_pnl,
        realized_pnl=portfolio.realized_pnl,
        total_return=total_return,
        return_percentage=return_percentage,
        holdings_count=len(holdings)
    )

@router.get("/holdings", response_model=List[HoldingResponse])
async def get_portfolio_holdings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed portfolio holdings"""
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        return []
    
    # Get holdings with bond details
    holdings = db.query(Holding, Bond).join(
        Bond, Holding.bond_id == Bond.id
    ).filter(Holding.portfolio_id == portfolio.id).all()
    
    holdings_response = []
    total_value = portfolio.total_value
    
    for holding, bond in holdings:
        return_percentage = ((holding.current_value - (holding.quantity * holding.avg_price)) / 
                           (holding.quantity * holding.avg_price) * 100) if holding.avg_price > 0 else 0
        
        weight = (holding.current_value / total_value * 100) if total_value > 0 else 0
        
        holdings_response.append(HoldingResponse(
            id=str(holding.id),
            bond_id=str(bond.id),
            bond_name=bond.name,
            bond_isin=bond.isin,
            issuer=bond.issuer,
            quantity=holding.quantity,
            avg_price=holding.avg_price,
            current_price=bond.current_price,
            current_value=holding.current_value,
            unrealized_pnl=holding.unrealized_pnl,
            return_percentage=return_percentage,
            weight=weight
        ))
    
    return holdings_response

@router.get("/risk-analysis", response_model=RiskAnalysis)
async def get_risk_analysis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio risk analysis"""
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get holdings and bonds
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    bond_ids = [h.bond_id for h in holdings]
    bonds = db.query(Bond).filter(Bond.id.in_(bond_ids)).all()
    
    if not holdings:
        return RiskAnalysis(
            portfolio_duration=0,
            modified_duration=0,
            convexity=0,
            var_1_day=0,
            var_10_day=0,
            beta=1.0,
            sharpe_ratio=0,
            volatility=0,
            max_drawdown=0
        )
    
    # Calculate risk metrics
    duration = analytics.calculate_portfolio_duration(holdings, bonds)
    avg_ytm = sum(b.ytm for b in bonds if b.ytm) / len([b for b in bonds if b.ytm]) if bonds else 6.5
    modified_duration = analytics.calculate_modified_duration(duration, avg_ytm)
    convexity = analytics.calculate_convexity(holdings, bonds)
    
    # Estimate volatility (simplified)
    volatility = 0.15  # 15% annual volatility estimate for bond portfolio
    
    # Calculate VaR
    var_1_day = abs(analytics.calculate_var(portfolio.total_value, volatility, 1))
    var_10_day = abs(analytics.calculate_var(portfolio.total_value, volatility, 10))
    
    # Get historical portfolio values for performance metrics
    portfolio_values = await get_historical_portfolio_values(str(portfolio.id), db)
    
    # Calculate Sharpe ratio and max drawdown
    returns = []
    if len(portfolio_values) > 1:
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
    
    sharpe_ratio = analytics.calculate_sharpe_ratio(returns)
    max_drawdown = analytics.calculate_max_drawdown(portfolio_values)
    
    return RiskAnalysis(
        portfolio_duration=duration,
        modified_duration=modified_duration,
        convexity=convexity,
        var_1_day=var_1_day,
        var_10_day=var_10_day,
        beta=0.8,  # Bonds typically have low beta
        sharpe_ratio=sharpe_ratio,
        volatility=volatility * 100,  # Convert to percentage
        max_drawdown=max_drawdown * 100  # Convert to percentage
    )

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio performance metrics"""
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get historical data
    portfolio_values = await get_historical_portfolio_values(str(portfolio.id), db, days)
    trades = db.query(Trade).filter(
        and_(
            Trade.user_id == current_user.id,
            Trade.status == "filled",
            Trade.executed_at >= datetime.utcnow() - timedelta(days=days)
        )
    ).all()
    
    if len(portfolio_values) < 2:
        return PerformanceMetrics(
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            calmar_ratio=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0
        )
    
    # Calculate returns
    returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
              for i in range(1, len(portfolio_values))]
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (365/days) - 1) * 100
    volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility
    
    sharpe_ratio = analytics.calculate_sharpe_ratio(returns)
    max_drawdown = analytics.calculate_max_drawdown(portfolio_values) * 100
    
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
    
    # Trade statistics
    winning_trades = [t for t in trades if t.total_amount > 0]  # Simplified
    losing_trades = [t for t in trades if t.total_amount < 0]   # Simplified
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.total_amount for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([abs(t.total_amount) for t in losing_trades]) if losing_trades else 0
    profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else 0
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor
    )

@router.get("/allocation/sector", response_model=List[SectorAllocation])
async def get_sector_allocation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio sector allocation"""
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        return []
    
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    bond_ids = [h.bond_id for h in holdings]
    bonds = db.query(Bond).filter(Bond.id.in_(bond_ids)).all()
    
    return analytics.get_sector_allocation(holdings, bonds)

@router.get("/allocation/maturity", response_model=List[MaturityAllocation])
async def get_maturity_allocation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio maturity allocation"""
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        return []
    
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    bond_ids = [h.bond_id for h in holdings]
    bonds = db.query(Bond).filter(Bond.id.in_(bond_ids)).all()
    
    return analytics.get_maturity_allocation(holdings, bonds)

@router.post("/rebalance", response_model=dict)
async def rebalance_portfolio(
    target_allocations: Dict[str, float],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rebalance portfolio based on target allocations"""
    
    if current_user.kyc_status != "verified":
        raise HTTPException(
            status_code=400,
            detail="KYC verification required for rebalancing"
        )
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Validate target allocations sum to 100%
    total_allocation = sum(target_allocations.values())
    if abs(total_allocation - 100) > 0.01:
        raise HTTPException(
            status_code=400,
            detail="Target allocations must sum to 100%"
        )
    
    # Get current holdings
    holdings = db.query(Holding, Bond).join(
        Bond, Holding.bond_id == Bond.id
    ).filter(Holding.portfolio_id == portfolio.id).all()
    
    current_allocations = {}
    total_value = portfolio.total_value
    
    for holding, bond in holdings:
        sector = analytics.get_sector_from_bond_type(bond.bond_type, bond.issuer)
        if sector not in current_allocations:
            current_allocations[sector] = 0
        current_allocations[sector] += holding.current_value / total_value * 100
    
    # Calculate rebalancing actions
    rebalance_actions = []
    for sector, target_pct in target_allocations.items():
        current_pct = current_allocations.get(sector, 0)
        difference = target_pct - current_pct
        
        if abs(difference) > 1:  # Only rebalance if difference > 1%
            action_type = "BUY" if difference > 0 else "SELL"
            amount = abs(difference) / 100 * total_value
            
            rebalance_actions.append({
                "sector": sector,
                "action": action_type,
                "amount": amount,
                "percentage_change": difference
            })
    
    return {
        "message": "Rebalancing analysis completed",
        "current_allocations": current_allocations,
        "target_allocations": target_allocations,
        "recommended_actions": rebalance_actions,
        "estimated_cost": len(rebalance_actions) * 100  # Simplified transaction cost
    }

@router.get("/stress-test", response_model=dict)
async def portfolio_stress_test(
    scenario: str = Query(default="interest_rate_shock", regex="^(interest_rate_shock|credit_spread_widening|market_crash)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform stress testing on portfolio"""
    
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    holdings = db.query(Holding, Bond).join(
        Bond, Holding.bond_id == Bond.id
    ).filter(Holding.portfolio_id == portfolio.id).all()
    
    if not holdings:
        return {
            "scenario": scenario,
            "current_value": 0,
            "stressed_value": 0,
            "loss_amount": 0,
            "loss_percentage": 0
        }
    
    current_value = portfolio.total_value
    stressed_value = current_value
    
    # Apply stress scenario
    if scenario == "interest_rate_shock":
        # +200 bps interest rate shock
        for holding, bond in holdings:
            if bond.duration:
                # Price change = -Duration × ΔYield
                price_change_pct = -bond.duration * 2.0 / 100  # 200 bps = 2%
                holding_loss = holding.current_value * price_change_pct
                stressed_value += holding_loss
    
    elif scenario == "credit_spread_widening":
        # +100 bps credit spread widening for corporate bonds
        for holding, bond in holdings:
            if bond.bond_type.lower() == "corporate" and bond.duration:
                price_change_pct = -bond.duration * 1.0 / 100  # 100 bps = 1%
                holding_loss = holding.current_value * price_change_pct
                stressed_value += holding_loss
    
    elif scenario == "market_crash":
        # 15% market decline
        stressed_value = current_value * 0.85
    
    loss_amount = current_value - stressed_value
    loss_percentage = loss_amount / current_value * 100 if current_value > 0 else 0
    
    return {
        "scenario": scenario,
        "scenario_description": get_scenario_description(scenario),
        "current_value": current_value,
        "stressed_value": stressed_value,
        "loss_amount": loss_amount,
        "loss_percentage": loss_percentage,
        "recommendations": get_stress_test_recommendations(scenario, loss_percentage)
    }

# Helper functions
async def get_historical_portfolio_values(portfolio_id: str, db: Session, days: int = 30) -> List[float]:
    """Get historical portfolio values"""
    # This would typically fetch from a time-series database
    # For now, simulate historical data
    base_value = 100000  # Base portfolio value
    values = []
    
    for i in range(days):
        # Simulate daily returns with some volatility
        daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return with 2% volatility
        if i == 0:
            values.append(base_value)
        else:
            values.append(values[-1] * (1 + daily_return))
    
    return values

def get_scenario_description(scenario: str) -> str:
    """Get description for stress test scenario"""
    descriptions = {
        "interest_rate_shock": "Interest rates increase by 200 basis points across all maturities",
        "credit_spread_widening": "Credit spreads widen by 100 basis points for corporate bonds",
        "market_crash": "Broad market decline of 15% affecting all asset classes"
    }
    return descriptions.get(scenario, "Unknown scenario")

def get_stress_test_recommendations(scenario: str, loss_percentage: float) -> List[str]:
    """Get recommendations based on stress test results"""
    recommendations = []
    
    if loss_percentage > 20:
        recommendations.append("Consider reducing portfolio duration to decrease interest rate sensitivity")
        recommendations.append("Diversify across different bond types and credit qualities")
    elif loss_percentage > 10:
        recommendations.append("Monitor interest rate exposure closely")
        recommendations.append("Consider adding some floating rate bonds")
    else:
        recommendations.append("Portfolio shows good resilience to the tested scenario")
    
    if scenario == "credit_spread_widening":
        recommendations.append("Consider reducing exposure to lower-rated corporate bonds")
        recommendations.append("Increase allocation to government securities")
    
    return recommendations