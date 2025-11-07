"""
FastAPI Backend for Legal Text Decoder
Serves trained models via REST API
"""
import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Legal text paragraph to evaluate")
    model_type: str = Field(default="baseline", description="Model type: 'baseline' or 'transformer'")


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str


class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    models_loaded: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Legal Text Decoder API",
    description="API for evaluating legal text readability (1-5 scale)",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/app/output')


def load_baseline_model():
    """Load baseline model (TF-IDF + LogisticRegression)"""
    model_path = os.path.join(OUTPUT_DIR, 'models', 'baseline_model.pkl')
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def load_transformer_model():
    """Load transformer model (HuBERT)"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_dir = os.path.join(OUTPUT_DIR, 'models', 'transformer_model')
        label_map_path = os.path.join(OUTPUT_DIR, 'models', 'label_mapping.json')
        
        if not os.path.exists(model_dir) or not os.path.exists(label_map_path):
            return None
        
        # Load label mapping
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_mapping': label_mapping,
            'device': device
        }
    except Exception as e:
        print(f"Warning: Could not load transformer model: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models
    
    print("Loading models...")
    
    # Load baseline model
    baseline = load_baseline_model()
    if baseline:
        models['baseline'] = baseline
        print("✓ Baseline model loaded")
    else:
        print("✗ Baseline model not found")
    
    # Load transformer model
    transformer = load_transformer_model()
    if transformer:
        models['transformer'] = transformer
        print("✓ Transformer model loaded")
    else:
        print("✗ Transformer model not found")
    
    if not models:
        print("WARNING: No models loaded! Train models first.")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    available = []
    loaded = list(models.keys())
    
    # Check which models are available
    baseline_path = os.path.join(OUTPUT_DIR, 'models', 'baseline_model.pkl')
    if os.path.exists(baseline_path):
        available.append('baseline')
    
    transformer_path = os.path.join(OUTPUT_DIR, 'models', 'transformer_model')
    if os.path.exists(transformer_path):
        available.append('transformer')
    
    return HealthResponse(
        status="healthy" if models else "no_models_loaded",
        models_available=available,
        models_loaded=loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict readability score for legal text
    
    Args:
        request: PredictionRequest with text and model_type
    
    Returns:
        PredictionResponse with prediction and probabilities
    """
    # Validate model type
    if request.model_type not in ['baseline', 'transformer']:
        raise HTTPException(status_code=400, detail="model_type must be 'baseline' or 'transformer'")
    
    # Check if model is loaded
    if request.model_type not in models:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{request.model_type}' not loaded. Available: {list(models.keys())}"
        )
    
    # Validate text
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Make prediction based on model type
    if request.model_type == 'baseline':
        return predict_baseline(request.text)
    else:
        return predict_transformer(request.text)


def predict_baseline(text: str) -> PredictionResponse:
    """Predict using baseline model"""
    model = models['baseline']
    
    # Get prediction
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    
    # Get class labels
    classes = model.classes_ if hasattr(model, 'classes_') else list(range(len(probabilities)))
    
    # Build probability dict
    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    
    # Get confidence (max probability)
    confidence = float(max(probabilities))
    
    return PredictionResponse(
        text=text[:200] + '...' if len(text) > 200 else text,
        prediction=str(prediction),
        confidence=confidence,
        probabilities=prob_dict,
        model_used='baseline'
    )


def predict_transformer(text: str) -> PredictionResponse:
    """Predict using transformer model"""
    import torch
    
    model_data = models['transformer']
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    device = model_data['device']
    id2label = model_data['label_mapping']['id2label']
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Build probability dict
    prob_dict = {
        id2label[str(i)]: float(probabilities[i].cpu().numpy())
        for i in range(len(probabilities))
    }
    
    # Get prediction and confidence
    prediction = id2label[str(predicted_class)]
    confidence = float(probabilities[predicted_class].cpu().numpy())
    
    return PredictionResponse(
        text=text[:200] + '...' if len(text) > 200 else text,
        prediction=prediction,
        confidence=confidence,
        probabilities=prob_dict,
        model_used='transformer'
    )


@app.get("/models")
async def list_models():
    """List available and loaded models"""
    available = []
    
    baseline_path = os.path.join(OUTPUT_DIR, 'models', 'baseline_model.pkl')
    if os.path.exists(baseline_path):
        available.append({
            'name': 'baseline',
            'type': 'TF-IDF + Logistic Regression',
            'loaded': 'baseline' in models
        })
    
    transformer_path = os.path.join(OUTPUT_DIR, 'models', 'transformer_model')
    if os.path.exists(transformer_path):
        available.append({
            'name': 'transformer',
            'type': 'HuBERT (Hungarian BERT)',
            'loaded': 'transformer' in models
        })
    
    return {
        'models': available,
        'total_loaded': len(models)
    }


if __name__ == "__main__":
    # Run server
    port = int(os.getenv('API_PORT', '8000'))
    uvicorn.run(app, host="0.0.0.0", port=port)
