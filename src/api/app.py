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


# ---------------------------------------------------------------------------
# Model architectures (mirrored from training script)
# ---------------------------------------------------------------------------

class Step1_BaselineModel(nn.Module):
    """Transformer + single linear classifier (CLS pooling)."""
    def __init__(self, transformer_model, num_classes=5):
        super().__init__()
        self.transformer = transformer_model
        hidden_size = transformer_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        out = type('Output', (), {'logits': logits})()
        if labels is not None:
            out.loss = nn.CrossEntropyLoss()(logits, labels)
        return out


class Step2_ExtendedModel(nn.Module):
    """Transformer + 2-layer adapter with BatchNorm + Dropout."""
    def __init__(self, transformer_model, num_classes=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.transformer = transformer_model
        trans_hidden = transformer_model.config.hidden_size
        self.adapter = nn.Sequential(
            nn.Linear(trans_hidden, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        adapted = self.adapter(pooled)
        logits = self.classifier(adapted)
        out = type('Output', (), {'logits': logits})()
        if labels is not None:
            out.loss = nn.CrossEntropyLoss()(logits, labels)
        return out


class Step3_AdvancedModel(nn.Module):
    """Transformer + attention pooling + deep adapter + gating."""
    def __init__(self, transformer_model, num_classes=5, hidden_dim=256, dropout=0.4, num_heads=4):
        super().__init__()
        self.transformer = transformer_model
        trans_hidden = transformer_model.config.hidden_size

        self.attention_pool = nn.MultiheadAttention(
            embed_dim=trans_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, trans_hidden))

        self.adapter_1 = nn.Sequential(
            nn.Linear(trans_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.adapter_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.adapter_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = outputs.last_hidden_state
        batch_size = hidden.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pool(query, hidden, hidden, key_padding_mask=~attention_mask.bool())
        pooled = pooled.squeeze(1)
        x = self.adapter_1(pooled)
        residual = x
        x = self.adapter_2(x)
        gate = torch.sigmoid(self.gate(x))
        x = gate * x + (1 - gate) * residual
        x = self.adapter_3(x)
        logits = self.classifier(x)
        out = type('Output', (), {'logits': logits})()
        if labels is not None:
            out.loss = nn.CrossEntropyLoss()(logits, labels)
        return out


class BalancedFinalModel(nn.Module):
    """Production-ready: transformer + mean pooling + balanced adapter."""
    def __init__(self, transformer_model, num_classes=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.transformer = transformer_model
        trans_hidden = transformer_model.config.hidden_size
        self.adapter = nn.Sequential(
            nn.Linear(trans_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        pooled = summed / counts
        adapted = self.adapter(pooled)
        logits = self.classifier(adapted)
        out = type('Output', (), {'logits': logits})()
        if labels is not None:
            out.loss = nn.CrossEntropyLoss()(logits, labels)
        return out


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
    """Load the best trained checkpoint (best_model.pt or best_*.pt)."""
    try:
        from transformers import AutoTokenizer, AutoModel

        models_dir = os.path.join(OUTPUT_DIR, 'models')
        best_overall_path = os.path.join(models_dir, 'best_overall.json')
        generic_best = os.path.join(models_dir, 'best_model.pt')

        best_checkpoint = None
        best_name = None

        if os.path.exists(best_overall_path):
            with open(best_overall_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                best_checkpoint = data.get('best_checkpoint')
                best_name = data.get('best_model_name')

        if not best_checkpoint or not os.path.exists(best_checkpoint):
            # Fallbacks: generic best, then any best_*.pt
            if os.path.exists(generic_best):
                best_checkpoint = generic_best
            else:
                pt_candidates = sorted(Path(models_dir).glob('best_*.pt'))
                if pt_candidates:
                    best_checkpoint = str(pt_candidates[0])
                    if not best_name:
                        best_name = pt_candidates[0].stem.replace('best_', '')

        if not best_checkpoint or not os.path.exists(best_checkpoint):
            print("No checkpoint found to load")
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint and label mapping
        checkpoint = torch.load(best_checkpoint, map_location=device)
        label2id = checkpoint.get('label2id', {})
        id2label = checkpoint.get('id2label', {})

        label_map_path = os.path.join(models_dir, 'label_mapping.json')
        if not label2id and os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                lm = json.load(f)
                label2id = lm.get('label2id', {})
                id2label = lm.get('id2label', {})

        num_classes = len(label2id) if label2id else 5

        # Choose base transformer
        base_model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')
        base_transformer = AutoModel.from_pretrained(base_model_name)

        # Map best_name to the correct architecture
        name_to_cls = {
            'Step1_Baseline': Step1_BaselineModel,
            'Step2_Extended': Step2_ExtendedModel,
            'Step3_Advanced': Step3_AdvancedModel,
            'Final_Balanced': BalancedFinalModel,
            'BalancedFinalModel': BalancedFinalModel,
        }
        # Normalize best_name if present
        model_cls = BalancedFinalModel
        if best_name:
            for key, cls in name_to_cls.items():
                if best_name.startswith(key) or best_name.endswith(key.lower()):
                    model_cls = cls
                    break

        model = model_cls(base_transformer, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Build mapping to use in responses
        if not id2label and label2id:
            id2label = {v: str(k) for k, v in label2id.items()}

        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_mapping': {'label2id': label2id, 'id2label': id2label},
            'device': device,
            'is_fusion': False,
            'checkpoint': best_checkpoint,
            'base_model_name': base_model_name,
            'model_name': best_name or model_cls.__name__
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
    
    models_dir = os.path.join(OUTPUT_DIR, 'models')
    generic_best = os.path.join(models_dir, 'best_model.pt')
    pt_candidates = list(Path(models_dir).glob('best_*.pt'))
    if os.path.exists(generic_best) or pt_candidates:
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
    """Predict using trained transformer checkpoint"""
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
    
    # Build probability dict (handle int vs str keys)
    prob_dict = {}
    for i in range(len(probabilities)):
        key = id2label.get(i) or id2label.get(str(i)) or str(i)
        prob_dict[key] = float(probabilities[i].cpu().numpy())
    
    # Get prediction and confidence
    prediction = id2label.get(predicted_class) or id2label.get(str(predicted_class)) or str(predicted_class)
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
