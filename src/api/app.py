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
import torch
import torch.nn as nn
import torch.nn.functional as F


# FusionModel class for transformer-based predictions
class FusionModel(nn.Module):
    """Transformer + standardized readability feature MLP fusion with mean pooling."""
    
    def __init__(self, transformer_model, num_classes=5, feature_dim=8, feat_hidden=32, hidden_dim=256, dropout=0.3, pooling='mean', use_coral=False):
        super().__init__()
        self.transformer = transformer_model
        self.feature_dim = feature_dim
        self.pooling = pooling
        self.num_classes = num_classes
        self.use_coral = use_coral
        transformer_hidden_size = transformer_model.config.hidden_size
        
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, feat_hidden),
            nn.LayerNorm(feat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.GELU()
        )
        
        self.fusion_fc = nn.Linear(transformer_hidden_size + feat_hidden, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            return summed / counts
        return hidden_states[:, 0, :]

    def forward(self, input_ids, attention_mask, readability_features=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden, attention_mask)
        
        if readability_features is not None:
            feat = self.feature_branch(readability_features)
        else:
            feat = torch.zeros(pooled.size(0), self.feature_branch[0].out_features, device=pooled.device)
        
        fused = torch.cat([pooled, feat], dim=1)
        x = self.dropout(F.gelu(self.fusion_fc(fused)))
        logits = self.classifier(x)
        
        output = type('Output', (), {'logits': logits})()
        return output


# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Legal text paragraph to evaluate")
    model_type: str = Field(default="transformer", description="Model type: 'transformer' only")


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


def load_transformer_model():
    """Load advanced FusionModel transformer"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_dir = os.path.join(OUTPUT_DIR, 'models', 'best_transformer_model')
        label_map_path = os.path.join(OUTPUT_DIR, 'models', 'label_mapping.json')
        fusion_config_path = os.path.join(model_dir, 'fusion_config.json')
        
        if not os.path.exists(model_dir):
            return None
        
        # Load label mapping
        label_mapping = {}
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)
        
        # Load fusion config
        fusion_config = {}
        if os.path.exists(fusion_config_path):
            with open(fusion_config_path, 'r', encoding='utf-8') as f:
                fusion_config = json.load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata to find base model name
        metadata = {}
        metadata_path = os.path.join(model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Load base transformer from HuggingFace (using model name from metadata)
        base_model_name = metadata.get('model_name', 'SZTAKI-HLT/hubert-base-cc')
        from transformers import AutoModel
        base_transformer = AutoModel.from_pretrained(base_model_name)
        
        # Create FusionModel wrapper
        fusion_model = FusionModel(
            base_transformer,
            num_classes=fusion_config.get('num_classes', 5),
            feature_dim=fusion_config.get('feature_dim', 8),
            hidden_dim=fusion_config.get('hidden_dim', 256),
            dropout=fusion_config.get('dropout', 0.3)
        )
        
        # Load saved weights
        model_weights_path = os.path.join(model_dir, 'pytorch_model.bin')
        if os.path.exists(model_weights_path):
            weights = torch.load(model_weights_path, map_location=device, weights_only=False)
            fusion_model.load_state_dict(weights, strict=False)
        
        fusion_model.to(device)
        fusion_model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        return {
            'model': fusion_model,
            'tokenizer': tokenizer,
            'label_mapping': label_mapping,
            'device': device,
            'is_fusion': True
        }
    except Exception as e:
        print(f"Warning: Could not load transformer model: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models
    
    print("Loading models...")
    
    # Load transformer model only
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
    transformer_path = os.path.join(OUTPUT_DIR, 'models', 'best_transformer_model')
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
    if request.model_type != 'transformer':
        raise HTTPException(status_code=400, detail="model_type must be 'transformer'")
    
    # Check if model is loaded
    if 'transformer' not in models:
        raise HTTPException(
            status_code=503, 
            detail="Transformer model not loaded. Train the model first."
        )
    
    # Validate text
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Make prediction with transformer
    return predict_transformer(request.text)


def predict_transformer(text: str) -> PredictionResponse:
    """Predict using FusionModel transformer"""
    model_data = models['transformer']
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    device = model_data['device']
    label_mapping = model_data.get('label_mapping', {})
    
    # Get id2label mapping or use defaults (1-5)
    if label_mapping and 'id2label' in label_mapping:
        id2label = label_mapping['id2label']
    else:
        # Default mapping for 5 classes (1-5 scale)
        id2label = {str(i): str(i+1) for i in range(5)}
    
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
        use_mixed = device.type == 'cuda'
        from torch import amp as torch_amp
        with torch_amp.autocast('cuda', enabled=use_mixed):
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
    
    transformer_path = os.path.join(OUTPUT_DIR, 'models', 'best_transformer_model')
    if os.path.exists(transformer_path):
        available.append({
            'name': 'transformer',
            'type': 'FusionModel (HuBERT + Readability Features)',
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
