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

        from src.services.evaluation import get_baseline_metrics
        metrics = get_baseline_metrics().get(model, {})
        
        # Generate Matplotlib Chart
        import base64
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io

        chart_base64 = None
        if metrics:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='none')
            ax.set_facecolor('none')
            
            labels = [k.replace('_', ' ').title() for k in metrics.keys()]
            values = [v * 100 for v in metrics.values()]
            
            # Professional unified color palette
            colors = ['#3b82f6', '#10b981', '#6366f1', '#8b5cf6']
            
            # Setup grid lines behind the bars for a premium look
            ax.set_axisbelow(True)
            ax.grid(axis='y', linestyle='--', alpha=0.2, color='#cbd5e1')
            
            # Sleek, thinner bars
            bars = ax.bar(labels, values, width=0.55, color=colors, alpha=0.95, edgecolor='none', zorder=3)
            
            ax.set_ylim(0, 115)
            ax.set_ylabel('Score (%)', fontsize=11, color='#94a3b8', labelpad=10)
            ax.set_title(f'{model.upper()} Benchmark Performance', fontsize=14, color='#f8fafc', pad=20, fontweight='bold')
            
            ax.tick_params(colors='#94a3b8', width=0, labelsize=10)
            
            # Minimalist spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_visible(False) # Hide left spine entirely for cleaner look
            
            # Elegant annotations
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, color='#f1f5f9', fontweight='600')
                            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, dpi=200) # Higher DPI for crispness
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # Attach recommendation object cleanly to the existing result
        result["recommendation"] = recommendation
        result["metrics"] = metrics
        if chart_base64:
            result["metrics_chart_base64"] = chart_base64
        
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

@app.on_event("startup")
async def startup_eval():
    """Trigger metric computation on startup in a separate thread if not already cached."""
    from src.evaluation.compute_metrics import compute_all_metrics
    import threading
    
    def run_eval():
        print("[Startup] Scanning for model performance metrics...")
        compute_all_metrics(force=False) # Only runs if cache is missing
    
    thread = threading.Thread(target=run_eval)
    thread.daemon = True
    thread.start()

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
