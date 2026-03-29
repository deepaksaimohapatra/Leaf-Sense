# Plant Knowledge Base Dictionary Structure
KNOWLEDGE_BASE = {
    "tomato": {
        "healthy": {
            "maintenance_tips": ["Prune regularly", "Keep leaves dry to prevent fungal growth"],
            "fertilizer_suggestions": ["Use NPK 5-10-10", "Add calcium-rich compost"],
            "watering_schedule": "Water deeply at base 1-2 times a week",
            "sunlight_requirements": "6-8 hours of direct sunlight",
            "yield_improvement_tips": ["Mulch around base", "Use stakes or cages for support"]
        },
        "diseases": {
            "general_disease": {
                "disease_name": "Tomato Disease (General)",
                "causes": ["Fungal or bacterial infection", "Poor air circulation", "Wet foliage"],
                "remedies": {
                    "organic": ["Neem oil spray", "Copper fungicide", "Remove affected leaves"],
                    "chemical": ["Chlorothalonil-based fungicide"]
                },
                "prevention_tips": ["Use drip irrigation", "Space plants adequately"]
            }
        }
    },
    "potato": {
        "healthy": {
            "maintenance_tips": ["Hilling the soil around stems", "Weed regularly"],
            "fertilizer_suggestions": ["Low nitrogen, high potassium and phosphorus"],
            "watering_schedule": "1-2 inches per week",
            "sunlight_requirements": "Full sun (at least 6 hours)",
            "yield_improvement_tips": ["Use certified disease-free seed potatoes"]
        },
        "diseases": {
            "general_disease": {
                "disease_name": "Potato Blight / Fungal Infection",
                "causes": ["Phytophthora infestans (fungus)", "Wet and humid weather"],
                "remedies": {
                    "organic": ["Copper fungicide", "Remove and destroy blighted plants"],
                    "chemical": ["Chlorothalonil", "Mancozeb"]
                },
                "prevention_tips": ["Crop rotation", "Use disease-resistant varieties"]
            }
        }
    },
    "apple": {
        "healthy": {
            "maintenance_tips": ["Dormant pruning in winter", "Thinning fruit early in season"],
            "fertilizer_suggestions": ["Nitrogen in early spring", "Boron if soil is deficient"],
            "watering_schedule": "Deeply once every 7-10 days for older trees, more for young",
            "sunlight_requirements": "Full sun (6+ hours)",
            "yield_improvement_tips": ["Cross-pollinate with different compatible varieties"]
        },
        "diseases": {
            "general_disease": {
                "disease_name": "Apple Scab / Fungal Disease",
                "causes": ["Venturia inaequalis (fungal)", "Prolonged leaf wetness"],
                "remedies": {
                    "organic": ["Liquid copper soap", "Sulfur powder"],
                    "chemical": ["Myclobutanil", "Captan fungicide"]
                },
                "prevention_tips": ["Rake and destroy fallen leaves", "Prune for better air flow"]
            }
        }
    }
}


class RecommendationEngine:
    @staticmethod
    def get_recommendation(plant_type: str, health_status: str):
        """
        Given the plant type and health status ('Healthy' or 'Diseased'),
        returns a structured recommendation dictionary.
        """
        plant_type = plant_type.lower()
        if plant_type not in KNOWLEDGE_BASE:
            return {"notice": "No specific recommendations available for this plant."}
            
        plant_info = KNOWLEDGE_BASE[plant_type]
        
        # In the existing predictor, labels are roughly Healthy / Diseased.
        if "healthy" in health_status.lower() and "unhealthy" not in health_status.lower() and "disease" not in health_status.lower():
            healthy_info = plant_info.get("healthy", {})
            return {
                "status": "Optimal",
                "guidelines": healthy_info
            }
        else:
            # We default to general disease logic since the classifier is binary
            disease_info = plant_info.get("diseases", {}).get("general_disease", {})
            return {
                "disease_name": disease_info.get("disease_name", "Unknown Disease"),
                "causes": disease_info.get("causes", []),
                "solutions": disease_info.get("remedies", {}),
                "prevention_tips": disease_info.get("prevention_tips", [])
            }
