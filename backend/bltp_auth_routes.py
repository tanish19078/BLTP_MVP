"""
BLTP Platform Backend - Authentication Routes
Handles user registration, login, KYC verification
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_
import re
import uuid
from datetime import datetime, timedelta
import httpx
from typing import Optional
import random
import string

# Import from main app (assumed to be in same package)
from main import (
    get_db, User, get_password_hash, verify_password, create_access_token,
    UserCreate, UserLogin, Token, redis_client, logger
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Additional Pydantic models for auth
from pydantic import BaseModel, validator

class KYCVerification(BaseModel):
    pan_number: str
    aadhaar_number: str
    bank_account: str
    ifsc_code: str
    
    @validator('pan_number')
    def validate_pan(cls, v):
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', v):
            raise ValueError('Invalid PAN format')
        return v
    
    @validator('ifsc_code')
    def validate_ifsc(cls, v):
        if not re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', v):
            raise ValueError('Invalid IFSC format')
        return v

class PasswordReset(BaseModel):
    email: str

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str
    phone: str
    kyc_status: str
    is_verified: bool
    created_at: str

# Utility functions for auth
def generate_otp() -> str:
    """Generate 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def generate_reset_token() -> str:
    """Generate password reset token"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

async def send_sms_otp(phone: str, otp: str) -> bool:
    """Send OTP via SMS - integrate with SMS provider like Twilio, MSG91"""
    try:
        # Placeholder for SMS API integration
        logger.info(f"SMS OTP {otp} sent to {phone}")
        return True
    except Exception as e:
        logger.error(f"SMS sending failed: {e}")
        return False

async def send_email_otp(email: str, otp: str) -> bool:
    """Send OTP via email - integrate with email service"""
    try:
        # Placeholder for email API integration
        logger.info(f"Email OTP {otp} sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        return False

async def verify_pan_with_income_tax(pan: str, name: str) -> dict:
    """Verify PAN with Income Tax Department API"""
    try:
        # Placeholder for actual PAN verification API
        # In production, integrate with NSDL PAN verification API
        
        # Simulate API response
        is_valid = len(pan) == 10 and pan.isalnum()
        
        return {
            "valid": is_valid,
            "name_match": True if is_valid else False,
            "status": "VALID" if is_valid else "INVALID"
        }
    except Exception as e:
        logger.error(f"PAN verification failed: {e}")
        return {"valid": False, "error": str(e)}

async def verify_bank_account(account_number: str, ifsc: str) -> dict:
    """Verify bank account using penny drop method"""
    try:
        # Placeholder for bank verification API
        # Integrate with services like Razorpay Fund Account Validation
        
        return {
            "valid": True,
            "account_holder_name": "Account Holder",
            "bank_name": "Sample Bank"
        }
    except Exception as e:
        logger.error(f"Bank verification failed: {e}")
        return {"valid": False, "error": str(e)}

# Authentication Routes
@router.post("/register", response_model=dict)
async def register(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Register new user with OTP verification"""
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists"
        )
    
    # Check PAN uniqueness
    existing_pan = db.query(User).filter(User.pan_number == user_data.pan_number).first()
    if existing_pan:
        raise HTTPException(
            status_code=400,
            detail="User with this PAN already exists"
        )
    
    # Validate PAN format
    if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', user_data.pan_number):
        raise HTTPException(
            status_code=400,
            detail="Invalid PAN format"
        )
    
    # Generate OTPs
    email_otp = generate_otp()
    sms_otp = generate_otp()
    
    # Store user data temporarily with OTPs
    temp_user_data = {
        "email": user_data.email,
        "password": get_password_hash(user_data.password),
        "full_name": user_data.full_name,
        "pan_number": user_data.pan_number,
        "phone": user_data.phone,
        "email_otp": email_otp,
        "sms_otp": sms_otp,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Store in Redis for 10 minutes
    redis_key = f"temp_user:{user_data.email}"
    redis_client.setex(redis_key, 600, str(temp_user_data))
    
    # Send OTPs
    background_tasks.add_task(send_email_otp, user_data.email, email_otp)
    background_tasks.add_task(send_sms_otp, user_data.phone, sms_otp)
    
    return {
        "message": "OTPs sent to email and phone. Please verify to complete registration.",
        "email": user_data.email,
        "phone": user_data.phone
    }

@router.post("/verify-registration", response_model=dict)
async def verify_registration(
    email: str,
    email_otp: str,
    sms_otp: str,
    db: Session = Depends(get_db)
):
    """Verify OTPs and complete user registration"""
    
    # Get temp user data
    redis_key = f"temp_user:{email}"
    temp_data = redis_client.get(redis_key)
    
    if not temp_data:
        raise HTTPException(
            status_code=400,
            detail="Registration session expired. Please register again."
        )
    
    temp_user = eval(temp_data)  # In production, use json.loads with proper serialization
    
    # Verify OTPs
    if temp_user["email_otp"] != email_otp or temp_user["sms_otp"] != sms_otp:
        raise HTTPException(
            status_code=400,
            detail="Invalid OTP"
        )
    
    # Create user
    db_user = User(
        email=temp_user["email"],
        hashed_password=temp_user["password"],
        full_name=temp_user["full_name"],
        pan_number=temp_user["pan_number"],
        phone=temp_user["phone"],
        is_verified=True
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Clean up temp data
    redis_client.delete(redis_key)
    
    # Create access token
    access_token = create_access_token(data={"sub": str(db_user.id)})
    
    return {
        "message": "Registration completed successfully",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(db_user.id),
            "email": db_user.email,
            "full_name": db_user.full_name,
            "kyc_status": db_user.kyc_status
        }
    }

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """User login with email and password"""
    
    user = db.query(User).filter(User.email == user_credentials.email).first()
    
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="Account is deactivated"
        )
    
    if not user.is_verified:
        raise HTTPException(
            status_code=401,
            detail="Email not verified. Please complete registration."
        )
    
    access_token = create_access_token(data={"sub": str(user.id)})
    
    # Log successful login
    redis_client.setex(f"user_session:{user.id}", 86400, access_token)
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/kyc-verification", response_model=dict)
async def kyc_verification(
    kyc_data: KYCVerification,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit KYC documents for verification"""
    
    if current_user.kyc_status == "verified":
        raise HTTPException(
            status_code=400,
            detail="KYC already verified"
        )
    
    # Verify PAN
    pan_verification = await verify_pan_with_income_tax(
        kyc_data.pan_number, 
        current_user.full_name
    )
    
    if not pan_verification.get("valid"):
        raise HTTPException(
            status_code=400,
            detail="PAN verification failed"
        )
    
    # Verify bank account
    bank_verification = await verify_bank_account(
        kyc_data.bank_account,
        kyc_data.ifsc_code
    )
    
    if not bank_verification.get("valid"):
        raise HTTPException(
            status_code=400,
            detail="Bank account verification failed"
        )
    
    # Update user KYC status
    current_user.kyc_status = "pending"
    db.commit()
    
    # Store KYC data (in production, encrypt sensitive data)
    kyc_redis_key = f"kyc_data:{current_user.id}"
    kyc_info = {
        "pan_number": kyc_data.pan_number,
        "aadhaar_number": kyc_data.aadhaar_number,
        "bank_account": kyc_data.bank_account,
        "ifsc_code": kyc_data.ifsc_code,
        "pan_verification": pan_verification,
        "bank_verification": bank_verification,
        "submitted_at": datetime.utcnow().isoformat()
    }
    
    redis_client.setex(kyc_redis_key, 2592000, str(kyc_info))  # 30 days
    
    return {
        "message": "KYC documents submitted successfully. Verification in progress.",
        "status": "pending",
        "estimated_verification_time": "24-48 hours"
    }

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    
    return UserProfile(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        phone=current_user.phone or "",
        kyc_status=current_user.kyc_status,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at.isoformat()
    )

@router.post("/password-reset", response_model=dict)
async def password_reset(
    reset_data: PasswordReset,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Initiate password reset"""
    
    user = db.query(User).filter(User.email == reset_data.email).first()
    if not user:
        # Don't reveal if user exists or not
        return {"message": "If the email exists, a reset link has been sent."}
    
    # Generate reset token
    reset_token = generate_reset_token()
    
    # Store reset token
    redis_client.setex(f"password_reset:{reset_token}", 3600, str(user.id))
    
    # Send reset email (background task)
    background_tasks.add_task(send_email_otp, user.email, f"Reset token: {reset_token}")
    
    return {"message": "If the email exists, a reset link has been sent."}

@router.post("/password-reset-confirm", response_model=dict)
async def password_reset_confirm(
    reset_data: PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    """Confirm password reset with token"""
    
    # Get user ID from reset token
    user_id = redis_client.get(f"password_reset:{reset_data.token}")
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired reset token"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=400,
            detail="User not found"
        )
    
    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    db.commit()
    
    # Delete reset token
    redis_client.delete(f"password_reset:{reset_data.token}")
    
    return {"message": "Password reset successfully"}

@router.post("/logout", response_model=dict)
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user and invalidate session"""
    
    # Remove session from Redis
    redis_client.delete(f"user_session:{current_user.id}")
    
    return {"message": "Logged out successfully"}