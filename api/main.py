import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference.health_classifier import get_health_classifier
import shutil
import os
import uuid
from PIL import Image
import io

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



@app.post("/diagnose-health")
async def api_diagnose_health(image: UploadFile = File(...), confirmed_plant: str = Form(...)):
    """
    Second step: Perform health diagnosis once plant type is confirmed.
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
    img = Image.open(io.BytesIO(contents))
    
    classifier = get_health_classifier()
    try:
        result = classifier.predict(img)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        print(f"Diagnosis Error: {str(e)}")
        return {
            "status": "error",
            "message": f"An error occurred during diagnosis: {str(e)}"
        }

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
