import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io

from src.services.model_loader import model_loader_instance
from src.services.predictor import PredictorService
from src.services.evaluation import EvaluationService
from src.services.recommendation_engine import RecommendationEngine

app = FastAPI(title="Plant Disease Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Preload models on startup
@app.on_event("startup")
async def load_models():
    print("Preloading ML Models...")
    model_loader_instance.preload_all_models()

@app.post("/diagnose-health")
async def api_diagnose_health(
    image: UploadFile = File(...), 
    confirmed_plant: str = Form(...),
    model: str = Form("resnet50")
):
    """
    Second step: Perform health diagnosis once plant type is confirmed.
    Optionally accepts a 'model' parameter (default: resnet50).
    """
    allowed_plants = ["apple", "tomato", "potato"]
    if confirmed_plant.lower() not in allowed_plants:
        return {
            "status": "error",
            "message": f"Sorry, the system currently only supports {', '.join(allowed_plants)}. You confirmed: {confirmed_plant}"
        }

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Process image directly from memory for diagnosis
    contents = await image.read()
    
    try:
        # Predict using selected model
        result = PredictorService.predict_health(contents, model_name=model)
        
        # Get Recommendations
        recommendation = RecommendationEngine.get_recommendation(
            plant_type=confirmed_plant, 
            health_status=result.get("prediction", "Healthy")
        )
        
        # Attach recommendation object cleanly to the existing result
        result["recommendation"] = recommendation
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        print(f"Diagnosis Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"An error occurred during diagnosis: {str(e)}"
        }

@app.post("/compare-models")
async def api_compare_models(image: UploadFile = File(...)):
    """
    Evaluate the image across all supported models and return metrics & predictions.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    contents = await image.read()
    
    try:
        results = EvaluationService.compare_models(contents)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
