from src.services.predictor import PredictorService

# Dummy baseline metrics since we evaluate on a single image live.
BASELINE_METRICS = {
    "cnn": {"accuracy": 0.85, "precision": 0.82, "recall": 0.87, "f1_score": 0.84},
    "resnet50": {"accuracy": 0.94, "precision": 0.95, "recall": 0.93, "f1_score": 0.94},
    "mobilenet": {"accuracy": 0.91, "precision": 0.90, "recall": 0.92, "f1_score": 0.91},
    "efficientnet": {"accuracy": 0.96, "precision": 0.96, "recall": 0.95, "f1_score": 0.95}
}

class EvaluationService:
    @staticmethod
    def compare_models(image):
        """
        Runs the image through all available models.
        Returns the predictions and baseline metrics for comparison.
        """
        models_to_test = ["cnn", "resnet50", "mobilenet", "efficientnet"]
        results = {}
        
        for model in models_to_test:
            try:
                # We reuse the predictor for each architecture
                pred_result = PredictorService.predict_health(image, model)
                metrics = BASELINE_METRICS.get(model, {})
                
                results[model] = {
                    "prediction_result": pred_result,
                    "metrics": metrics
                }
            except Exception as e:
                results[model] = {
                    "error": str(e)
                }
                
        return {"models": results}
