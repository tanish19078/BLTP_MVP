"""
BLTP Platform Backend - WebSocket & AI Integration Routes
Handles real-time data feeds, AI-powered recommendations, and chat functionality
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import openai
import httpx

# Import from main app
from main import (
    get_db, get_current_user, User, Bond, Portfolio, Holding, Trade,
    manager, redis_client, logger, MarketData
)

# AI and WebSocket routers
ws_router = APIRouter()
ai_router = APIRouter(prefix="/ai", tags=["AI & Recommendations"])

# Pydantic models for AI
from pydantic import BaseModel

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]
    timestamp: str

class RecommendationRequest(BaseModel):
    risk_tolerance: str  # conservative, moderate, aggressive
    investment_amount: float
    investment_horizon: int  # in months
    preferences: Optional[Dict[str, Any]] = None

class BondRecommendation(BaseModel):
    bond_id: str
    bond_name: str
    isin: str
    issuer: str
    current_price: float
    ytm: float
    rating: str
    recommendation_score: float
    reasons: List[str]
    risk_level: str

class MarketInsight(BaseModel):
    title: str
    content: str
    impact: str  # positive, negative, neutral
    confidence: float
    timestamp: str

# AI Recommendation Engine
class BondRecommendationEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'coupon_rate', 'ytm', 'duration', 'face_value', 
            'days_to_maturity', 'rating_score', 'current_price'
        ]
        
    async def initialize_model(self, db: Session):
        """Initialize or load the recommendation model"""
        try:
            # Try to load existing model
            self.model = joblib.load('bond_recommendation_model.pkl')
            self.scaler = joblib.load('bond_scaler.pkl')
            logger.info("Loaded existing recommendation model")
        except:
            # Train new model if none exists
            await self.train_model(db)
    
    async def train_model(self, db: Session):
        """Train the bond recommendation model"""
        try:
            # Get bond data
            bonds = db.query(Bond).all()
            if len(bonds) < 10:
                logger.warning("Insufficient data to train recommendation model")
                return
            
            # Create training dataset
            data = []
            for bond in bonds:
                if bond.maturity_date and bond.current_price:
                    days_to_maturity = (bond.maturity_date - datetime.utcnow()).days
                    rating_score = self.convert_rating_to_score(bond.rating)
                    
                    data.append({
                        'coupon_rate': bond.coupon_rate,
                        'ytm': bond.ytm or 6.5,
                        'duration': bond.duration or 3.0,
                        'face_value': bond.face_value,
                        'days_to_maturity': days_to_maturity,
                        'rating_score': rating_score,
                        'current_price': bond.current_price,
                        'target': self.calculate_bond_score(bond)
                    })
            
            if len(data) < 5:
                return
            
            df = pd.DataFrame(data)
            
            # Prepare features and target
            X = df[self.feature_columns]
            y = df['target']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Save model
            joblib.dump(self.model, 'bond_recommendation_model.pkl')
            joblib.dump(self.scaler, 'bond_scaler.pkl')
            
            logger.info("Trained new recommendation model")
            
        except Exception as e:
            logger.error(f"Error training recommendation model: {e}")
    
    def convert_rating_to_score(self, rating: str) -> float:
        """Convert credit rating to numerical score"""
        rating_map = {
            'AAA': 10, 'AA+': 9, 'AA': 8, 'AA-': 7,
            'A+': 6, 'A': 5, 'A-': 4,
            'BBB+': 3, 'BBB': 2, 'BBB-': 1,
            'BB': 0, 'B': -1, 'CCC': -2
        }
        return rating_map.get(rating, 2)
    
    def calculate_bond_score(self, bond: Bond) -> float:
        """Calculate bond attractiveness score"""
        score = 0
        
        # YTM factor
        if bond.ytm:
            score += min(bond.ytm / 10, 1.0) * 30
        
        # Rating factor
        rating_score = self.convert_rating_to_score(bond.rating)
        score += (rating_score / 10) * 25
        
        # Duration factor (prefer moderate duration)
        if bond.duration:
            duration_score = 1 - abs(bond.duration - 4) / 10
            score += max(duration_score, 0) * 20
        
        # Liquidity factor (simplified)
        score += np.random.uniform(0, 25)
        
        return min(score, 100)
    
    async def get_recommendations(
        self, 
        user_profile: Dict,
        available_bonds: List[Bond],
        count: int = 5
    ) -> List[BondRecommendation]:
        """Get personalized bond recommendations"""
        
        if not self.model or not available_bonds:
            return []
        
        try:
            recommendations = []
            
            for bond in available_bonds:
                if not bond.is_tradable or not bond.current_price:
                    continue
                
                # Prepare features
                days_to_maturity = (bond.maturity_date - datetime.utcnow()).days if bond.maturity_date else 1000
                rating_score = self.convert_rating_to_score(bond.rating)
                
                features = np.array([[
                    bond.coupon_rate,
                    bond.ytm or 6.5,
                    bond.duration or 3.0,
                    bond.face_value,
                    days_to_maturity,
                    rating_score,
                    bond.current_price
                ]])
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Predict recommendation score
                base_score = self.model.predict(features_scaled)[0]
                
                # Adjust based on user profile
                adjusted_score = self.adjust_score_for_user(base_score, bond, user_profile)
                
                # Generate reasons
                reasons = self.generate_recommendation_reasons(bond, user_profile)
                
                # Determine risk level
                risk_level = self.determine_risk_level(bond)
                
                recommendations.append(BondRecommendation(
                    bond_id=str(bond.id),
                    bond_name=bond.name,
                    isin=bond.isin,
                    issuer=bond.issuer,
                    current_price=bond.current_price,
                    ytm=bond.ytm or 0,
                    rating=bond.rating,
                    recommendation_score=adjusted_score,
                    reasons=reasons,
                    risk_level=risk_level
                ))
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
            return recommendations[:count]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def adjust_score_for_user(self, base_score: float, bond: Bond, user_profile: Dict) -> float:
        """Adjust recommendation score based on user profile"""
        risk_tolerance = user_profile.get('risk_tolerance', 'moderate')
        investment_horizon = user_profile.get('investment_horizon', 24)  # months
        
        score = base_score
        
        # Risk tolerance adjustment
        rating_score = self.convert_rating_to_score(bond.rating)
        
        if risk_tolerance == 'conservative':
            if rating_score >= 7:  # AA- and above
                score *= 1.2
            elif rating_score < 4:  # Below A-
                score *= 0.7
        elif risk_tolerance == 'aggressive':
            if rating_score < 4:  # Below A- (higher yield potential)
                score *= 1.3
            elif rating_score >= 8:  # AA and above
                score *= 0.9
        
        # Investment horizon adjustment
        if bond.maturity_date:
            months_to_maturity = (bond.maturity_date - datetime.utcnow()).days / 30
            horizon_match = 1 - abs(months_to_maturity - investment_horizon) / max(months_to_maturity, investment_horizon)
            score *= (0.8 + 0.4 * horizon_match)  # Adjust between 0.8 and 1.2
        
        return min(score, 100)
    
    def generate_recommendation_reasons(self, bond: Bond, user_profile: Dict) -> List[str]:
        """Generate reasons for recommendation"""
        reasons = []
        
        if bond.ytm and bond.ytm > 7:
            reasons.append(f"Attractive yield of {bond.ytm:.2f}%")
        
        rating_score = self.convert_rating_to_score(bond.rating)
        if rating_score >= 7:
            reasons.append(f"High credit rating: {bond.rating}")
        
        if bond.bond_type.lower() == 'government':
            reasons.append("Government backing provides security")
        
        if bond.duration and 2 <= bond.duration <= 5:
            reasons.append("Moderate duration reduces interest rate risk")
        
        return reasons[:3]  # Return top 3 reasons
    
    def determine_risk_level(self, bond: Bond) -> str:
        """Determine risk level of the bond"""
        rating_score = self.convert_rating_to_score(bond.rating)
        
        if rating_score >= 8:
            return "Low"
        elif rating_score >= 5:
            return "Medium"
        else:
            return "High"

# AI Chat Engine
class BondChatEngine:
    def __init__(self):
        self.openai_api_key = "your-openai-api-key"  # Set from environment
        self.system_prompt = """
        You are an expert bond trading assistant for the Indian bond market. 
        You help users understand bond investments, analyze market conditions, 
        and provide educational content about fixed income securities.
        
        Key areas of expertise:
        - Indian government and corporate bonds
        - Yield analysis and interest rate trends
        - Credit risk assessment
        - Portfolio diversification strategies
        - Market timing and economic indicators
        
        Always provide accurate, helpful information while noting that this is 
        educational content and not personalized financial advice.
        """
    
    async def process_chat_message(
        self, 
        message: str, 
        user_context: Dict,
        db: Session
    ) -> ChatResponse:
        """Process chat message and generate AI response"""
        
        try:
            # Get user portfolio context
            portfolio_context = await self.get_portfolio_context(user_context.get('user_id'), db)
            
            # Get market context
            market_context = await self.get_market_context(db)
            
            # Prepare context for AI
            full_context = f"""
            User Portfolio Context: {portfolio_context}
            Current Market Context: {market_context}
            User Message: {message}
            """
            
            # For demo purposes, generate a simple response
            # In production, integrate with OpenAI or other LLM
            response_text = await self.generate_ai_response(message, full_context)
            
            # Generate suggestions
            suggestions = self.generate_suggestions(message)
            
            return ChatResponse(
                response=response_text,
                suggestions=suggestions,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return ChatResponse(
                response="I apologize, but I'm having trouble processing your request right now. Please try again.",
                suggestions=["What are government bonds?", "How do I analyze bond yields?", "Show me my portfolio"],
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_portfolio_context(self, user_id: str, db: Session) -> str:
        """Get user portfolio context for AI"""
        if not user_id:
            return "No portfolio information available"
        
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
            if not portfolio:
                return "No portfolio found"
            
            holdings_count = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).count()
            
            return f"""
            Portfolio Value: ₹{portfolio.total_value:,.2f}
            Cash Balance: ₹{portfolio.cash_balance:,.2f}
            Holdings: {holdings_count} bonds
            Unrealized P&L: ₹{portfolio.unrealized_pnl:,.2f}
            Realized P&L: ₹{portfolio.realized_pnl:,.2f}
            """
        except Exception as e:
            logger.error(f"Error getting portfolio context: {e}")
            return "Portfolio information unavailable"
    
    async def get_market_context(self, db: Session) -> str:
        """Get current market context"""
        try:
            # Get recent market data
            recent_data = db.query(MarketData).filter(
                MarketData.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ).order_by(MarketData.timestamp.desc()).limit(5).all()
            
            if not recent_data:
                return "Market data unavailable"
            
            context = "Recent Market Data:\n"
            for data in recent_data:
                context += f"- {data.symbol}: ₹{data.price} ({data.change:+.2f}%)\n"
            
            return context
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return "Market context unavailable"
    
    async def generate_ai_response(self, message: str, context: str) -> str:
        """Generate AI response (simplified version)"""
        # This is a simplified response generator
        # In production, integrate with OpenAI GPT or other LLM
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['portfolio', 'holdings', 'investments']):
            return """Based on your portfolio, I can see you have several bond holdings. Here are some insights:

1. **Diversification**: Your portfolio shows good diversification across different bond types
2. **Risk Management**: Consider your overall duration exposure for interest rate risk
3. **Yield Optimization**: Look for opportunities to enhance yield while maintaining your risk profile

Would you like me to analyze any specific aspect of your portfolio in more detail?"""
        
        elif any(word in message_lower for word in ['yield', 'ytm', 'interest rate']):
            return """Yield analysis is crucial for bond investing. Here's what to consider:

**Current Market Environment:**
- 10-year G-Sec yields are around 7.2-7.4%
- Corporate bond spreads vary by rating (50-200 bps over G-Sec)
- RBI policy rates influence overall yield environment

**Key Metrics:**
- **Yield to Maturity (YTM)**: Total return if held to maturity
- **Current Yield**: Annual coupon / Current price
- **Duration**: Price sensitivity to interest rate changes

Would you like me to explain how to calculate or compare yields for specific bonds?"""
        
        elif any(word in message_lower for word in ['risk', 'credit', 'rating']):
            return """Risk management in bond investing involves several factors:

**Credit Risk:**
- Government bonds: Lowest risk (sovereign guarantee)
- AAA Corporate: Very low default risk
- Lower ratings: Higher risk but potentially higher returns

**Interest Rate Risk:**
- Longer duration = Higher sensitivity to rate changes
- Consider laddering maturities to manage this risk

**Liquidity Risk:**
- Some bonds may be harder to sell before maturity
- Government bonds typically most liquid

**Recommendations:**
- Diversify across ratings and sectors
- Match duration to your investment horizon
- Keep some allocation to liquid instruments

Need help assessing risk for specific bonds?"""
        
        elif any(word in message_lower for word in ['buy', 'sell', 'trade']):
            return """For bond trading decisions, consider these factors:

**Before Buying:**
- Analyze credit quality and rating
- Compare YTM with similar maturity bonds
- Check minimum investment requirements
- Assess liquidity and trading volumes

**Before Selling:**
- Consider tax implications (STCG vs LTCG)
- Evaluate current market price vs purchase price
- Check if holding to maturity is better
- Consider reinvestment options

**Market Timing:**
- Interest rate outlook affects bond prices
- Economic indicators impact credit spreads
- RBI policy meetings can cause volatility

Would you like help analyzing any specific trade you're considering?"""
        
        else:
            return """I'm here to help you with bond investing questions! I can assist with:

- **Portfolio Analysis**: Review your holdings and suggest improvements
- **Bond Selection**: Help find bonds matching your criteria
- **Risk Assessment**: Evaluate credit and interest rate risks
- **Yield Analysis**: Compare different bonds and calculate returns
- **Market Insights**: Explain current market conditions
- **Trading Strategies**: Discuss timing and execution

What specific aspect of bond investing would you like to explore?"""
    
    def generate_suggestions(self, message: str) -> List[str]:
        """Generate relevant follow-up suggestions"""
        message_lower = message.lower()
        
        if 'portfolio' in message_lower:
            return [
                "Analyze my bond allocation",
                "Show me portfolio risk metrics",
                "Suggest rebalancing strategies"
            ]
        elif 'yield' in message_lower:
            return [
                "Compare current yields with historical data",
                "Explain yield curve implications",
                "Find high-yield opportunities"
            ]
        elif 'risk' in message_lower:
            return [
                "Perform stress test on portfolio",
                "Explain credit rating meanings",
                "Show duration analysis"
            ]
        else:
            return [
                "What are the best bonds to buy now?",
                "How do I analyze bond risks?",
                "Show me my portfolio performance"
            ]

# Initialize engines
recommendation_engine = BondRecommendationEngine()
chat_engine = BondChatEngine()

# WebSocket endpoint
@ws_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time data"""
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "subscribe_market":
                # Subscribe to market data updates
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "data": {"subscribed_to": "market_data"}
                }))
            
            elif message_type == "subscribe_portfolio":
                # Subscribe to portfolio updates
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
                    if portfolio:
                        portfolio_data = {
                            "total_value": portfolio.total_value,
                            "unrealized_pnl": portfolio.unrealized_pnl,
                            "cash_balance": portfolio.cash_balance
                        }
                        await websocket.send_text(json.dumps({
                            "type": "portfolio_update",
                            "data": portfolio_data
                        }))
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"User {user_id} disconnected")

# AI Routes
@ai_router.post("/chat", response_model=ChatResponse)
async def ai_chat(
    chat_request: ChatMessage,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI-powered chat for bond investment queries"""
    
    user_context = {
        "user_id": str(current_user.id),
        "kyc_status": current_user.kyc_status,
        "context": chat_request.context
    }
    
    response = await chat_engine.process_chat_message(
        chat_request.message,
        user_context,
        db
    )
    
    return response

@ai_router.post("/recommendations", response_model=List[BondRecommendation])
async def get_ai_recommendations(
    request: RecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-powered bond recommendations"""
    
    if current_user.kyc_status != "verified":
        raise HTTPException(
            status_code=400,
            detail="KYC verification required for recommendations"
        )
    
    # Initialize model if not done already
    if not recommendation_engine.model:
        await recommendation_engine.initialize_model(db)
    
    # Get available bonds
    available_bonds = db.query(Bond).filter(Bond.is_tradable == True).all()
    
    # Prepare user profile
    user_profile = {
        "risk_tolerance": request.risk_tolerance,
        "investment_amount": request.investment_amount,
        "investment_horizon": request.investment_horizon,
        "preferences": request.preferences or {}
    }
    
    # Get recommendations
    recommendations = await recommendation_engine.get_recommendations(
        user_profile,
        available_bonds,
        count=5
    )
    
    return recommendations

@ai_router.get("/market-insights", response_model=List[MarketInsight])
async def get_market_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-generated market insights"""
    
    insights = []
    
    try:
        # Get recent market data
        recent_data = db.query(MarketData).filter(
            MarketData.timestamp >= datetime.utcnow() - timedelta(days=1)
        ).order_by(MarketData.timestamp.desc()).limit(20).all()
        
        # Generate insights based on market data
        if recent_data:
            # Interest rate trend analysis
            nifty_data = [d for d in recent_data if d.symbol == "NIFTY"]
            if len(nifty_data) >= 2:
                change = nifty_data[0].price - nifty_data[1].price
                impact = "positive" if change > 0 else "negative"
                
                insights.append(MarketInsight(
                    title="Equity Market Movement Impact",
                    content=f"Nifty has moved {change:+.2f} points, which typically has an inverse correlation with bond prices. {'Rising' if change > 0 else 'Falling'} equity markets may lead to {'decreased' if change > 0 else 'increased'} demand for bonds.",
                    impact=impact,
                    confidence=0.75,
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        # Add general market insights
        insights.append(MarketInsight(
            title="RBI Policy Outlook",
            content="With inflation showing signs of moderation, the RBI may pause rate hikes in the near term. This could be positive for bond prices, especially in the longer duration segment.",
            impact="positive",
            confidence=0.8,
            timestamp=datetime.utcnow().isoformat()
        ))
        
        insights.append(MarketInsight(
            title="Corporate Bond Spreads",
            content="Credit spreads in the corporate bond market remain stable. AAA-rated corporate bonds are trading at attractive spreads over government securities, offering good risk-adjusted returns.",
            impact="positive",
            confidence=0.85,
            timestamp=datetime.utcnow().isoformat()
        ))
        
        insights.append(MarketInsight(
            title="Liquidity Conditions",
            content="Banking system liquidity remains in surplus mode. This supportive liquidity environment is positive for bond market sentiment and could keep yields range-bound.",
            impact="positive",
            confidence=0.9,
            timestamp=datetime.utcnow().isoformat()
        ))
        
    except Exception as e:
        logger.error(f"Error generating market insights: {e}")
        # Return default insight if error occurs
        insights.append(MarketInsight(
            title="Market Analysis",
            content="Market conditions are being analyzed. Please check back for updated insights.",
            impact="neutral",
            confidence=0.5,
            timestamp=datetime.utcnow().isoformat()
        ))
    
    return insights[:5]  # Return top 5 insights

@ai_router.post("/analyze-bond/{bond_id}", response_model=dict)
async def analyze_bond_with_ai(
    bond_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI-powered analysis of a specific bond"""
    
    bond = db.query(Bond).filter(Bond.id == bond_id).first()
    if not bond:
        raise HTTPException(status_code=404, detail="Bond not found")
    
    try:
        # Calculate various metrics
        days_to_maturity = (bond.maturity_date - datetime.utcnow()).days if bond.maturity_date else None
        years_to_maturity = days_to_maturity / 365.25 if days_to_maturity else None
        
        # Credit analysis
        credit_score = recommendation_engine.convert_rating_to_score(bond.rating)
        credit_quality = "High" if credit_score >= 7 else "Medium" if credit_score >= 4 else "Low"
        
        # Yield analysis
        risk_free_rate = 7.2  # Approximate 10Y G-Sec yield
        credit_spread = (bond.ytm or 0) - risk_free_rate if bond.ytm else 0
        
        # Price analysis
        price_to_par = bond.current_price / bond.face_value if bond.current_price and bond.face_value else 1
        premium_discount = "Premium" if price_to_par > 1.01 else "Discount" if price_to_par < 0.99 else "At Par"
        
        # Generate AI analysis
        analysis = {
            "bond_overview": {
                "name": bond.name,
                "issuer": bond.issuer,
                "isin": bond.isin,
                "bond_type": bond.bond_type,
                "rating": bond.rating,
                "credit_quality": credit_quality
            },
            "financial_metrics": {
                "current_price": bond.current_price,
                "face_value": bond.face_value,
                "coupon_rate": bond.coupon_rate,
                "ytm": bond.ytm,
                "duration": bond.duration,
                "years_to_maturity": round(years_to_maturity, 2) if years_to_maturity else None,
                "credit_spread": round(credit_spread, 2) if credit_spread else None
            },
            "investment_thesis": {
                "strengths": [],
                "risks": [],
                "recommendation": "HOLD",  # Default recommendation
                "target_price": None,
                "suitability": []
            }
        }
        
        # Generate strengths
        if bond.ytm and bond.ytm > 7.5:
            analysis["investment_thesis"]["strengths"].append("Attractive yield above market average")
        
        if credit_score >= 8:
            analysis["investment_thesis"]["strengths"].append("High credit quality with low default risk")
        
        if bond.bond_type.lower() == "government":
            analysis["investment_thesis"]["strengths"].append("Government backing provides security")
        
        if bond.duration and 2 <= bond.duration <= 5:
            analysis["investment_thesis"]["strengths"].append("Moderate duration reduces interest rate sensitivity")
        
        # Generate risks
        if credit_score < 4:
            analysis["investment_thesis"]["risks"].append("Lower credit rating increases default risk")
        
        if bond.duration and bond.duration > 7:
            analysis["investment_thesis"]["risks"].append("High duration increases interest rate risk")
        
        if premium_discount == "Premium":
            analysis["investment_thesis"]["risks"].append("Trading at premium to face value")
        
        # Generate recommendation
        total_score = 0
        if bond.ytm and bond.ytm > 7:
            total_score += 2
        if credit_score >= 7:
            total_score += 2
        if bond.duration and 2 <= bond.duration <= 5:
            total_score += 1
        
        if total_score >= 4:
            analysis["investment_thesis"]["recommendation"] = "BUY"
        elif total_score >= 2:
            analysis["investment_thesis"]["recommendation"] = "HOLD"
        else:
            analysis["investment_thesis"]["recommendation"] = "SELL"
        
        # Generate suitability
        if bond.bond_type.lower() == "government" or credit_score >= 8:
            analysis["investment_thesis"]["suitability"].append("Conservative investors")
        
        if bond.ytm and bond.ytm > 7:
            analysis["investment_thesis"]["suitability"].append("Income-focused investors")
        
        if years_to_maturity and years_to_maturity <= 3:
            analysis["investment_thesis"]["suitability"].append("Short-term investors")
        elif years_to_maturity and years_to_maturity > 7:
            analysis["investment_thesis"]["suitability"].append("Long-term investors")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing bond: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

# Background task to train models periodically
async def retrain_models():
    """Background task to retrain AI models"""
    while True:
        try:
            # Wait 24 hours before retraining
            await asyncio.sleep(86400)
            
            # Get database session
            db = next(get_db())
            
            # Retrain recommendation model
            await recommendation_engine.train_model(db)
            
            logger.info("AI models retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
        finally:
            if 'db' in locals():
                db.close()
                