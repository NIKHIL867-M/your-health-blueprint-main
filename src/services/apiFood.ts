// src/services/apiFood.ts

export interface FoodPrediction {
  class_id: number;
  food: string;
  confidence: number;
}

export async function predictFood(file: File): Promise<FoodPrediction> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8002/predict-food", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Food prediction API error");
  }

  return res.json();
}
