# backend/ai3_food_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import json
from keras.models import load_model

app = FastAPI(title="Food Image Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ML Model + Labels
# -----------------------------
MODEL = load_model("hybrid_best.keras")

with open("ai3.json", "r") as f:
    IDX_TO_NAME = json.load(f)

# -----------------------------
# Load nutrition database (NEW)
# -----------------------------
with open("ai4.json", "r") as f:
    FOOD_DB = json.load(f)

IMG_SIZE = (224, 224)  # change if your model uses different size


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict-food")
async def predict_food(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        content = await file.read()
        arr = preprocess_image(content)

        # Model prediction
        preds = MODEL.predict(arr)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # Get predicted food name
        food_name = IDX_TO_NAME.get(str(class_idx), "unknown")

        # NEW: find nutrition from food db
        nutrition = FOOD_DB.get(food_name)

        # Prepare response object
        result = {
            "class_id": class_idx,
            "food": food_name,
            "confidence": confidence,
        }

        # Attach nutrition only if found
        if nutrition:
            result["nutrition"] = {
                "calories": nutrition.get("calories"),
                "protein": nutrition.get("protein"),
                "carbs": nutrition.get("carbs"),
                "fat": nutrition.get("fat"),
                "serving_size": nutrition.get("serving_size"),
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("ai3_food_api:app", host="0.0.0.0", port=8002, reload=True)
