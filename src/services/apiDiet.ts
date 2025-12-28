// src/services/apiDiet.ts

// -------------------------
// User Profile Payload
// -------------------------
export interface UserProfilePayload {
  age: number;
  gender: "male" | "female" | "other";
  height_cm: number;
  weight_kg: number;

  activity_level:
    | "sedentary"
    | "light"
    | "moderate"
    | "active"
    | "very_active"
    | "athlete";

  goal: "fat_loss" | "muscle_gain" | "maintenance";

  goal_intensity?: "conservative" | "moderate" | "aggressive";
  weekly_rate_kg?: number;
}

// -------------------------
// Food item payload (sent to diet analysis)
// -------------------------
export interface FoodItemPayload {
  food_name: string;
  estimated_calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
}

// -------------------------
// FOOD NUTRITION (NEW)
// Matches backend ai4.json values
// -------------------------
export interface FoodNutrition {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  serving_size?: string;
}

// -------------------------
// PREDICTION RESULT (NEW)
// Returned by food image API
// nutrition is optional
// -------------------------
export interface FoodPrediction {
  class_id: number;
  food: string;
  confidence: number;
  nutrition?: FoodNutrition;
}

// -------------------------
// DIET ANALYSIS API
// -------------------------
export async function analyzeDiet(
  profile: UserProfilePayload,
  foods: FoodItemPayload[] | null
) {
  const body = {
    user_profile: profile,
    food_intake: foods,
  };

  const res = await fetch("http://localhost:8001/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error("Diet API error");
  }

  return res.json();
}
