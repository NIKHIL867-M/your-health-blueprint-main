"""
Adaptive Intelligent Nutrition & Body Recomposition AI System
Complete Production-Ready Implementation
Fixed and Enhanced for AI Dietician & Calorie Intelligence Engine
"""

# ================ IMPORTS & CONFIGURATION ================
import numpy as np
import pandas as pd # type: ignore
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import logging
from abc import ABC, abstractmethod
import hashlib
import uuid
import warnings

# FastAPI imports for web interface
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel, Field, field_validator, ConfigDict # type: ignore
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis # type: ignore
from redis import Redis # type: ignore

# ================ SETUP LOGGING ================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================ DATABASE MODELS ================
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Biometrics
    age = Column(Integer)
    gender = Column(String)  # 'male', 'female', 'other'
    height_cm = Column(Float)
    weight_kg = Column(Float)
    body_fat_percentage = Column(Float, nullable=True)
    activity_level = Column(String)  # 'sedentary', 'light', 'moderate', 'active', 'very_active'
    
    # Goals
    primary_goal = Column(String)  # 'fat_loss', 'muscle_gain', 'maintenance'
    goal_intensity = Column(String)  # 'conservative', 'moderate', 'aggressive'
    target_weight_kg = Column(Float, nullable=True)
    weekly_rate_kg = Column(Float, default=0.5)  # kg per week
    
    # Current plan
    current_calories = Column(Float)
    current_protein_g = Column(Float)
    current_fat_g = Column(Float)
    current_carbs_g = Column(Float)
    
    # Adaptive history
    adaptation_history = Column(JSON, default=list)
    feedback_history = Column(JSON, default=list)
    
    # Settings
    preferences = Column(JSON, default=dict)
    constraints = Column(JSON, default=dict)

class WeeklyCheckin(Base):
    __tablename__ = "weekly_checkins"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    week_start = Column(DateTime, nullable=False)
    week_end = Column(DateTime, nullable=False)
    
    # Progress data
    weight_kg = Column(Float)
    adherence_score = Column(Float)  # 0-100%
    training_frequency = Column(Integer)  # days trained this week
    energy_level = Column(String, nullable=True)  # 'low', 'medium', 'high'
    hunger_level = Column(String, nullable=True)  # 'low', 'medium', 'high'
    
    # System analysis
    expected_weight_kg = Column(Float)
    weight_delta_kg = Column(Float)
    calories_adjustment = Column(Float, default=0)
    macros_adjustment = Column(JSON, default=dict)
    
    # User feedback
    user_feedback = Column(JSON, default=dict)

class DailyLog(Base):
    __tablename__ = "daily_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Actual intake
    calories_consumed = Column(Float)
    protein_g = Column(Float)
    fat_g = Column(Float)
    carbs_g = Column(Float)
    
    # Planned intake
    planned_calories = Column(Float)
    planned_protein_g = Column(Float)
    planned_fat_g = Column(Float)
    planned_carbs_g = Column(Float)
    
    # Training
    trained_today = Column(Boolean, default=False)
    training_type = Column(String, nullable=True)

# ================ ENUMS & DATA MODELS ================
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class Goal(str, Enum):
    FAT_LOSS = "fat_loss"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"

class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"        # Little or no exercise
    LIGHT = "light"               # Light exercise 1-3 days/week
    MODERATE = "moderate"         # Moderate exercise 3-5 days/week
    ACTIVE = "active"             # Hard exercise 6-7 days/week
    VERY_ACTIVE = "very_active"   # Very hard exercise & physical job
    ATHLETE = "athlete"           # Professional athlete (2.0+ multiplier)

class GoalIntensity(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class EnergyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# ================ INPUT/OUTPUT MODELS ================
class FoodItem(BaseModel):
    """Model for food intake data"""
    food_name: str
    estimated_calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    
    @field_validator('estimated_calories', 'protein_g', 'carbs_g', 'fat_g')
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Nutrition values cannot be negative")
        return v

class UserProfile(BaseModel):
    """Model for user profile input"""
    age: int = Field(ge=13, le=120, description="Age in years")
    gender: Gender
    height_cm: float = Field(ge=100, le=250, description="Height in centimeters")
    weight_kg: float = Field(ge=30, le=300, description="Weight in kilograms")
    activity_level: ActivityLevel
    goal: Goal
    goal_intensity: GoalIntensity = GoalIntensity.MODERATE
    body_fat_percentage: Optional[float] = Field(None, ge=5, le=60, description="Optional body fat percentage")
    weekly_rate_kg: float = Field(0.5, ge=0.1, le=1.0, description="Weekly weight change goal in kg")

class DieticianRequest(BaseModel):
    """Complete request for AI Dietician analysis"""
    user_profile: UserProfile
    food_intake: Optional[List[FoodItem]] = None
    consistency_data: Optional[Dict] = None
    progress_data: Optional[Dict] = None

class DieticianResponse(BaseModel):
    """Structured response from AI Dietician"""
    model_config = ConfigDict(protected_namespaces=())
    
    # Metabolic Summary
    metabolic_summary: Dict[str, float]
    
    # Daily Targets
    daily_targets: Dict[str, float]
    
    # Food Analysis (optional)
    food_analysis: Optional[Dict[str, Any]] = None
    
    # Smart Recommendations
    smart_recommendations: List[str]
    
    # Additional Insights
    additional_insights: Optional[Dict[str, Any]] = None

# ================ CORE AI ENGINE ================
class AIDieticianEngine:
    """
    AI Dietician & Calorie Intelligence Engine
    Combines scientific metabolic formulas with adaptive intelligence
    """
    
    # Activity multipliers for TDEE
    ACTIVITY_MULTIPLIERS = {
        ActivityLevel.SEDENTARY: 1.2,
        ActivityLevel.LIGHT: 1.375,
        ActivityLevel.MODERATE: 1.55,
        ActivityLevel.ACTIVE: 1.725,
        ActivityLevel.VERY_ACTIVE: 1.9,
        ActivityLevel.ATHLETE: 2.0
    }
    
    # Safety constraints
    MIN_CALORIES = 1200
    MAX_DEFICIT_PERCENT = 25
    MAX_SURPLUS_PERCENT = 15
    MIN_PROTEIN_PER_KG = 1.6
    MAX_PROTEIN_PER_KG = 2.5
    
    def __init__(self):
        self.adaptation_history = []
        
    def calculate_bmr(self, user_profile: UserProfile) -> float:
        """
        Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation
        Scientifically accepted formula for BMR calculation
        """
        weight = user_profile.weight_kg
        height = user_profile.height_cm
        age = user_profile.age
        
        if user_profile.gender == Gender.MALE:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        elif user_profile.gender == Gender.FEMALE:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        else:  # OTHER - use average of male and female
            bmr_male = (10 * weight) + (6.25 * height) - (5 * age) + 5
            bmr_female = (10 * weight) + (6.25 * height) - (5 * age) - 161
            bmr = (bmr_male + bmr_female) / 2
        
        return round(bmr, 1)
    
    def calculate_tdee(self, bmr: float, activity_level: ActivityLevel) -> float:
        """
        Calculate Total Daily Energy Expenditure using correct activity multipliers
        """
        if activity_level not in self.ACTIVITY_MULTIPLIERS:
            raise ValueError(f"Unknown activity level: {activity_level}")
        
        tdee = bmr * self.ACTIVITY_MULTIPLIERS[activity_level]
        return round(tdee, 1)
    
    def calculate_calorie_target(self, tdee: float, user_profile: UserProfile) -> Dict[str, float]:
        """
        Generate calorie targets based on goal with safety constraints
        No extreme or unsafe values
        """
        goal = user_profile.goal
        intensity = user_profile.goal_intensity
        weekly_rate = user_profile.weekly_rate_kg
        
        # Weekly calorie adjustment needed for desired weight change
        # 1 kg of fat ≈ 7700 kcal
        weekly_calorie_adjustment = weekly_rate * 7700
        daily_adjustment = weekly_calorie_adjustment / 7
        
        # Apply goal direction with intensity modifiers
        intensity_multipliers = {
            GoalIntensity.CONSERVATIVE: 0.7,
            GoalIntensity.MODERATE: 1.0,
            GoalIntensity.AGGRESSIVE: 1.3
        }
        
        intensity_mult = intensity_multipliers.get(intensity, 1.0)
        adjusted_daily = daily_adjustment * intensity_mult
        
        if goal == Goal.FAT_LOSS:
            target_calories = tdee - adjusted_daily
        elif goal == Goal.MUSCLE_GAIN:
            target_calories = tdee + adjusted_daily
        else:  # Maintenance
            target_calories = tdee
        
        # Apply safety constraints
        original_target = target_calories
        
        # 1. Minimum calorie constraint
        target_calories = max(target_calories, self.MIN_CALORIES)
        
        # 2. Maximum deficit/surplus constraints
        if goal == Goal.FAT_LOSS:
            max_deficit = tdee * (self.MAX_DEFICIT_PERCENT / 100)
            min_calories = tdee - max_deficit
            target_calories = max(target_calories, min_calories)
        elif goal == Goal.MUSCLE_GAIN:
            max_surplus = tdee * (self.MAX_SURPLUS_PERCENT / 100)
            max_calories = tdee + max_surplus
            target_calories = min(target_calories, max_calories)
        
        # Calculate adjustment percentage
        adjustment_percent = ((target_calories - tdee) / tdee) * 100
        
        return {
            'maintenance_calories': tdee,
            'goal_calories': round(target_calories, 1),
            'adjustment_percent': round(adjustment_percent, 1),
            'weekly_rate_kg': weekly_rate,
            'is_extreme': original_target != target_calories,
            'safety_applied': {
                'min_calories_violated': original_target < self.MIN_CALORIES,
                'max_adjustment_violated': (
                    (goal == Goal.FAT_LOSS and (tdee - original_target) > tdee * (self.MAX_DEFICIT_PERCENT / 100)) or
                    (goal == Goal.MUSCLE_GAIN and (original_target - tdee) > tdee * (self.MAX_SURPLUS_PERCENT / 100))
                )
            }
        }
    
    def calculate_macronutrients(self, calories: float, user_profile: UserProfile) -> Dict[str, float]:
        """
        Generate intelligent macronutrient split
        - Protein based on body weight & goal
        - Fat based on healthy percentage range
        - Carbohydrates as remaining calories
        - Ensure total macros match calorie target
        """
        weight = user_profile.weight_kg
        goal = user_profile.goal
        
        # Protein calculation (context-aware)
        if goal == Goal.FAT_LOSS:
            # Higher protein during fat loss to preserve muscle
            protein_per_kg = 2.2
        elif goal == Goal.MUSCLE_GAIN:
            protein_per_kg = 2.0
        else:  # Maintenance
            protein_per_kg = 1.8
        
        # Apply bounds
        protein_per_kg = max(self.MIN_PROTEIN_PER_KG, min(protein_per_kg, self.MAX_PROTEIN_PER_KG))
        protein_g = weight * protein_per_kg
        protein_calories = protein_g * 4
        
        # Fat calculation (percentage-based)
        if goal == Goal.FAT_LOSS:
            fat_percentage = 0.25  # 25% of calories from fat
        elif goal == Goal.MUSCLE_GAIN:
            fat_percentage = 0.28  # 28% of calories from fat
        else:
            fat_percentage = 0.27  # 27% of calories from fat
        
        fat_calories = calories * fat_percentage
        fat_g = fat_calories / 9
        
        # Ensure minimum fat for hormone health
        min_fat_g = 0.6 * weight  # Minimum 0.6g per kg for essential fatty acids
        fat_g = max(fat_g, min_fat_g)
        
        # Recalculate fat calories after ensuring minimum
        fat_calories = fat_g * 9
        
        # Carbohydrates (remaining calories)
        remaining_calories = calories - protein_calories - fat_calories
        carbs_g = remaining_calories / 4
        
        # Recalculate to ensure totals match
        total_calories = (protein_g * 4) + (fat_g * 9) + (carbs_g * 4)
        
        # Minor adjustment if needed (due to fat minimum enforcement)
        if abs(total_calories - calories) > 5:
            calorie_diff = calories - total_calories
            carbs_g += calorie_diff / 4
        
        # Calculate percentages
        protein_percent = (protein_g * 4) / calories * 100
        fat_percent = (fat_g * 9) / calories * 100
        carbs_percent = (carbs_g * 4) / calories * 100
        
        return {
            'protein_g': round(protein_g, 1),
            'fat_g': round(fat_g, 1),
            'carbs_g': round(carbs_g, 1),
            'protein_percent': round(protein_percent, 1),
            'fat_percent': round(fat_percent, 1),
            'carbs_percent': round(carbs_percent, 1),
            'calories': round(calories, 1)
        }
    
    def analyze_food_intake(self, food_items: List[FoodItem], 
                          daily_targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze food intake against daily targets
        - Compare consumed vs target calories
        - Show surplus or deficit
        - Highlight macro imbalances
        """
        if not food_items:
            return None
        
        # Calculate totals from food intake
        total_calories = sum(item.estimated_calories for item in food_items)
        total_protein = sum(item.protein_g for item in food_items)
        total_carbs = sum(item.carbs_g for item in food_items)
        total_fat = sum(item.fat_g for item in food_items)
        
        # Compare with targets
        calorie_diff = total_calories - daily_targets['calories']
        protein_diff = total_protein - daily_targets['protein_g']
        carbs_diff = total_carbs - daily_targets['carbs_g']
        fat_diff = total_fat - daily_targets['fat_g']
        
        # Determine status
        if abs(calorie_diff) < 50:
            status = "on target"
        elif calorie_diff > 0:
            status = f"surplus (+{round(calorie_diff)} kcal)"
        else:
            status = f"deficit ({round(calorie_diff)} kcal)"
        
        # Macro balance assessment
        macro_imbalances = []
        
        if abs(protein_diff) > 20:
            direction = "over" if protein_diff > 0 else "under"
            macro_imbalances.append(f"Protein: {direction} by {abs(round(protein_diff))}g")
        
        if abs(carbs_diff) > 30:
            direction = "over" if carbs_diff > 0 else "under"
            macro_imbalances.append(f"Carbs: {direction} by {abs(round(carbs_diff))}g")
        
        if abs(fat_diff) > 15:
            direction = "over" if fat_diff > 0 else "under"
            macro_imbalances.append(f"Fat: {direction} by {abs(round(fat_diff))}g")
        
        macro_balance = "good" if len(macro_imbalances) == 0 else "needs improvement"
        
        return {
            'calories_consumed': round(total_calories, 1),
            'protein_consumed': round(total_protein, 1),
            'carbs_consumed': round(total_carbs, 1),
            'fat_consumed': round(total_fat, 1),
            'calorie_diff': round(calorie_diff, 1),
            'protein_diff': round(protein_diff, 1),
            'carbs_diff': round(carbs_diff, 1),
            'fat_diff': round(fat_diff, 1),
            'status': status,
            'macro_balance': macro_balance,
            'macro_imbalances': macro_imbalances,
            'food_count': len(food_items)
        }
    
    def generate_recommendations(self, user_profile: UserProfile, 
                               metabolic_summary: Dict,
                               daily_targets: Dict,
                               food_analysis: Optional[Dict] = None,
                               consistency_data: Optional[Dict] = None) -> List[str]:
        """
        Generate SMART recommendations based on analysis
        - What to eat more/less
        - Next-day adjustment (if needed)
        - One simple, actionable suggestion
        """
        recommendations = []
        goal = user_profile.goal
        intensity = user_profile.goal_intensity
        
        # 1. Goal-specific recommendations
        if goal == Goal.FAT_LOSS:
            recommendations.append(f"Aim for a {intensity} deficit of ~{abs(metabolic_summary['adjustment_percent'])}% below maintenance.")
            recommendations.append("Prioritize protein at each meal to preserve muscle mass while losing fat.")
            recommendations.append("Consider time-restricted eating (12-14 hour window) if it fits your lifestyle.")
        elif goal == Goal.MUSCLE_GAIN:
            recommendations.append(f"Aim for a {intensity} surplus of ~{metabolic_summary['adjustment_percent']}% above maintenance.")
            recommendations.append("Distribute protein evenly across 3-5 meals throughout the day.")
            recommendations.append("Time carbs around workouts for optimal energy and recovery.")
        else:  # Maintenance
            recommendations.append("Focus on hitting your calorie target within ±100 kcal daily.")
            recommendations.append("Maintain consistent protein intake to support body composition.")
            recommendations.append("Listen to hunger/fullness cues to maintain weight naturally.")
        
        # 2. Food analysis specific recommendations
        if food_analysis:
            if food_analysis['status'].startswith('surplus'):
                recommendations.append(f"Reduce portion sizes slightly tomorrow to offset today's {abs(food_analysis['calorie_diff'])} kcal surplus.")
            elif food_analysis['status'].startswith('deficit'):
                recommendations.append(f"Add a protein-rich snack tomorrow to make up today's {abs(food_analysis['calorie_diff'])} kcal deficit.")
            
            if food_analysis['macro_imbalances']:
                for imbalance in food_analysis['macro_imbalances']:
                    if "Protein" in imbalance and "under" in imbalance:
                        recommendations.append("Add a protein source like Greek yogurt, chicken, or tofu to your next meal.")
                    elif "Carbs" in imbalance and "over" in imbalance:
                        recommendations.append("Swap some refined carbs for vegetables at your next meal.")
                    elif "Fat" in imbalance and "over" in imbalance:
                        recommendations.append("Use cooking spray instead of oil for your next meal preparation.")
        
        # 3. Adaptive recommendations based on consistency
        if consistency_data:
            adherence = consistency_data.get('adherence_score', 0)
            if adherence < 70:
                recommendations.append("Focus on consistency: Aim to hit within 10% of your calorie target for 5+ days this week.")
            elif adherence > 90:
                recommendations.append("Excellent consistency! Consider if your current plan feels sustainable long-term.")
        
        # 4. One simple, actionable suggestion (always included)
        simple_suggestions = [
            "Drink 500ml water 30 minutes before each main meal.",
            "Add one extra serving of vegetables to your largest meal today.",
            "Take a 10-minute walk after your next meal.",
            "Get 7-8 hours of sleep tonight for optimal metabolic health.",
            "Measure your food portions for the next 3 days to improve accuracy."
        ]
        
        import random
        recommendations.append(f"Today's simple action: {random.choice(simple_suggestions)}")
        
        return recommendations
    
    def adaptive_intelligence(self, user_profile: UserProfile, 
                            progress_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Adaptive intelligence for plateau detection and adjustments
        - Detect plateaus and suggest small calorie changes
        - Keep recommendations realistic and sustainable
        """
        if not progress_data:
            return {'adjustment_needed': False, 'reason': 'Insufficient progress data'}
        
        # Extract progress metrics
        weeks_tracked = progress_data.get('weeks_tracked', 0)
        avg_weekly_change = progress_data.get('avg_weekly_change_kg', 0)
        expected_change = progress_data.get('expected_weekly_change_kg', user_profile.weekly_rate_kg)
        adherence = progress_data.get('adherence_score', 80)
        
        adjustments = {
            'adjustment_needed': False,
            'calorie_adjustment': 0,
            'reason': '',
            'confidence': 'low'
        }
        
        # Only adjust after sufficient tracking
        if weeks_tracked >= 2:
            deviation_ratio = abs(avg_weekly_change) / abs(expected_change) if expected_change != 0 else 0
            
            # Plateau detection
            if abs(avg_weekly_change) < 0.1 and adherence > 80:
                adjustments['adjustment_needed'] = True
                if user_profile.goal == Goal.FAT_LOSS:
                    adjustments['calorie_adjustment'] = -100
                    adjustments['reason'] = f"Plateau detected: no weight change for {weeks_tracked} weeks despite good adherence"
                else:
                    adjustments['calorie_adjustment'] = 100
                    adjustments['reason'] = f"Plateau detected: no weight change for {weeks_tracked} weeks despite good adherence"
                adjustments['confidence'] = 'medium'
            
            # Too fast/slow progress
            elif deviation_ratio < 0.5 and adherence > 80:
                adjustments['adjustment_needed'] = True
                if user_profile.goal == Goal.FAT_LOSS:
                    adjustments['calorie_adjustment'] = -150
                    adjustments['reason'] = f"Progress slower than expected: {abs(avg_weekly_change):.2f}kg/week vs target {abs(expected_change):.2f}kg/week"
                else:
                    adjustments['calorie_adjustment'] = 150
                    adjustments['reason'] = f"Progress slower than expected: {avg_weekly_change:.2f}kg/week vs target {expected_change:.2f}kg/week"
                adjustments['confidence'] = 'medium'
            
            elif deviation_ratio > 1.5 and adherence > 80:
                adjustments['adjustment_needed'] = True
                if user_profile.goal == Goal.FAT_LOSS:
                    adjustments['calorie_adjustment'] = 100
                    adjustments['reason'] = f"Progress faster than expected: {abs(avg_weekly_change):.2f}kg/week vs target {abs(expected_change):.2f}kg/week"
                else:
                    adjustments['calorie_adjustment'] = -100
                    adjustments['reason'] = f"Progress faster than expected: {avg_weekly_change:.2f}kg/week vs target {expected_change:.2f}kg/week"
                adjustments['confidence'] = 'medium'
        
        return adjustments
    
    def process_request(self, request: DieticianRequest) -> DieticianResponse:
        """
        Main processing pipeline for AI Dietician requests
        Combines all components into structured response
        """
        user_profile = request.user_profile
        
        # 1. Calculate metabolic metrics
        bmr = self.calculate_bmr(user_profile)
        tdee = self.calculate_tdee(bmr, user_profile.activity_level)
        calorie_targets = self.calculate_calorie_target(tdee, user_profile)
        
        # 2. Calculate daily targets
        daily_targets = self.calculate_macronutrients(
            calorie_targets['goal_calories'], 
            user_profile
        )
        
        # 3. Analyze food intake if provided
        food_analysis = None
        if request.food_intake:
            food_analysis = self.analyze_food_intake(request.food_intake, daily_targets)
        
        # 4. Adaptive intelligence for progress adjustments
        adaptive_insights = None
        if request.progress_data:
            adaptive_insights = self.adaptive_intelligence(user_profile, request.progress_data)
        
        # 5. Generate recommendations
        recommendations = self.generate_recommendations(
            user_profile, 
            calorie_targets,
            daily_targets,
            food_analysis,
            request.consistency_data
        )
        
        # 6. Prepare metabolic summary
        metabolic_summary = {
            'BMR': bmr,
            'TDEE': tdee,
            'maintenance_calories': calorie_targets['maintenance_calories'],
            'goal_calories': calorie_targets['goal_calories'],
            'adjustment_percent': calorie_targets['adjustment_percent'],
            'weekly_rate_kg': calorie_targets['weekly_rate_kg']
        }
        
        # 7. Prepare response
        response_data = {
            'metabolic_summary': metabolic_summary,
            'daily_targets': {
                'calories': daily_targets['calories'],
                'protein_g': daily_targets['protein_g'],
                'carbs_g': daily_targets['carbs_g'],
                'fat_g': daily_targets['fat_g'],
                'protein_percent': daily_targets['protein_percent'],
                'carbs_percent': daily_targets['carbs_percent'],
                'fat_percent': daily_targets['fat_percent']
            },
            'food_analysis': food_analysis,
            'smart_recommendations': recommendations
        }
        
        if adaptive_insights:
            response_data['additional_insights'] = {
                'adaptive_adjustments': adaptive_insights,
                'next_checkpoint': '1_week',
                'monitoring_suggestions': [
                    "Track weight at same time daily (morning, after bathroom, before eating)",
                    "Take progress photos weekly in consistent lighting",
                    "Monitor energy levels and hunger cues"
                ]
            }
        
        return DieticianResponse(**response_data)

# ================ FASTAPI APPLICATION ================
app = FastAPI(
    title="AI Dietician & Calorie Intelligence Engine",
    description="Scientific nutrition guidance with adaptive intelligence",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine
dietician_engine = AIDieticianEngine()

@app.post("/analyze", response_model=DieticianResponse)
async def analyze_nutrition(request: DieticianRequest):
    """
    Main endpoint for AI Dietician analysis
    Takes user profile and optional food intake
    Returns structured nutrition guidance
    """
    try:
        logger.info(f"Processing request for user: {request.user_profile}")
        
        response = dietician_engine.process_request(request)
        
        logger.info("Analysis completed successfully")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Dietician Engine",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/quick-analysis")
async def quick_analysis(
    age: int = Query(..., ge=13, le=120),
    gender: Gender = Query(...),
    height_cm: float = Query(..., ge=100, le=250),
    weight_kg: float = Query(..., ge=30, le=300),
    activity_level: ActivityLevel = Query(...),
    goal: Goal = Query(...),
    goal_intensity: GoalIntensity = Query(GoalIntensity.MODERATE)
):
    """
    Quick analysis endpoint without full request body
    """
    user_profile = UserProfile(
        age=age,
        gender=gender,
        height_cm=height_cm,
        weight_kg=weight_kg,
        activity_level=activity_level,
        goal=goal,
        goal_intensity=goal_intensity
    )
    
    request = DieticianRequest(user_profile=user_profile)
    return dietician_engine.process_request(request)

# ================ FORMATTED OUTPUT GENERATOR ================
def format_response_for_cli(response: DieticianResponse) -> str:
    """
    Format the response in the required CLI format
    """
    output = []
    
    # 1. METABOLIC SUMMARY
    output.append("=" * 50)
    output.append("1) METABOLIC SUMMARY")
    output.append("-" * 50)
    output.append(f"• BMR: {response.metabolic_summary['BMR']} kcal")
    output.append(f"• TDEE: {response.metabolic_summary['TDEE']} kcal")
    if 'adjustment_percent' in response.metabolic_summary:
        direction = "deficit" if response.metabolic_summary['adjustment_percent'] < 0 else "surplus"
        output.append(f"• Goal adjustment: {abs(response.metabolic_summary['adjustment_percent'])}% {direction}")
    
    # 2. DAILY TARGETS
    output.append("\n" + "=" * 50)
    output.append("2) DAILY TARGETS")
    output.append("-" * 50)
    output.append(f"• Goal calories: {response.daily_targets['calories']} kcal")
    output.append(f"• Protein: {response.daily_targets['protein_g']} g ({response.daily_targets['protein_percent']}%)")
    output.append(f"• Carbs: {response.daily_targets['carbs_g']} g ({response.daily_targets['carbs_percent']}%)")
    output.append(f"• Fat: {response.daily_targets['fat_g']} g ({response.daily_targets['fat_percent']}%)")
    
    # 3. FOOD ANALYSIS (if exists)
    if response.food_analysis:
        output.append("\n" + "=" * 50)
        output.append("3) FOOD ANALYSIS")
        output.append("-" * 50)
        output.append(f"• Calories consumed: {response.food_analysis['calories_consumed']} kcal")
        output.append(f"• Status: {response.food_analysis['status']}")
        output.append(f"• Macro balance: {response.food_analysis['macro_balance']}")
        
        if response.food_analysis['macro_imbalances']:
            output.append("• Macro imbalances detected:")
            for imbalance in response.food_analysis['macro_imbalances']:
                output.append(f"  - {imbalance}")
    
    # 4. SMART RECOMMENDATIONS
    output.append("\n" + "=" * 50)
    output.append("4) SMART RECOMMENDATIONS")
    output.append("-" * 50)
    for i, rec in enumerate(response.smart_recommendations, 1):
        output.append(f"{i}. {rec}")
    
    # Additional insights if available
    if response.additional_insights:
        output.append("\n" + "=" * 50)
        output.append("ADDITIONAL INSIGHTS")
        output.append("-" * 50)
        
        adj = response.additional_insights.get('adaptive_adjustments', {})
        if adj.get('adjustment_needed'):
            output.append(f"• Adaptive adjustment suggested: {adj['calorie_adjustment']} kcal")
            output.append(f"• Reason: {adj['reason']}")
            output.append(f"• Confidence: {adj['confidence']}")
        
        monitoring = response.additional_insights.get('monitoring_suggestions', [])
        if monitoring:
            output.append("• Monitoring suggestions:")
            for suggestion in monitoring:
                output.append(f"  - {suggestion}")
    
    output.append("\n" + "=" * 50)
    output.append("Note: Always consult with a healthcare professional")
    output.append("before making significant dietary changes.")
    output.append("=" * 50)
    
    return "\n".join(output)

# ================ EXAMPLE USAGE ================
if __name__ == "__main__":
    # Example 1: Basic analysis without food intake
    print("EXAMPLE 1: BASIC ANALYSIS")
    print("-" * 50)
    
    user_profile = UserProfile(
        age=30,
        gender=Gender.MALE,
        height_cm=180,
        weight_kg=85,
        activity_level=ActivityLevel.MODERATE,
        goal=Goal.FAT_LOSS,
        goal_intensity=GoalIntensity.MODERATE,
        weekly_rate_kg=0.5
    )
    
    request = DieticianRequest(user_profile=user_profile)
    response = dietician_engine.process_request(request)
    print(format_response_for_cli(response))
    
    # Example 2: Analysis with food intake
    print("\n\nEXAMPLE 2: ANALYSIS WITH FOOD INTAKE")
    print("-" * 50)
    
    food_intake = [
        FoodItem(
            food_name="Grilled Chicken Breast",
            estimated_calories=230,
            protein_g=43,
            carbs_g=0,
            fat_g=5
        ),
        FoodItem(
            food_name="Brown Rice",
            estimated_calories=215,
            protein_g=5,
            carbs_g=45,
            fat_g=2
        ),
        FoodItem(
            food_name="Mixed Vegetables",
            estimated_calories=120,
            protein_g=4,
            carbs_g=20,
            fat_g=3
        )
    ]
    
    request2 = DieticianRequest(
        user_profile=user_profile,
        food_intake=food_intake
    )
    
    response2 = dietician_engine.process_request(request2)
    print(format_response_for_cli(response2))