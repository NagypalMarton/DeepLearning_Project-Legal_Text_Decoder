# Deep Learning Class (VITMMA19) – Legal Text Decoder

## Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Nagypál Márton Péter
- **Aiming for +1 Mark**: Yes

## Solution Description

### Problem Statement
The goal of this project is to automatically evaluate the readability of Hungarian legal documents (Terms of Service, contracts, policies, etc.) on a **1–5 scale** using modern Natural Language Processing techniques. This addresses a practical need in legal technology where understanding document complexity is crucial for accessibility and compliance.

### Model Architecture
The solution implements a **two-tier model approach**:

1. **Baseline Model**: Lightweight HuBERT-based transformer for sequence classification with standard fine-tuning. This serves as a reference baseline for evaluating advanced techniques.

2. **Advanced FusionModel**: Combines:
   - **Transformer embeddings** from HuBERT (contextual understanding)
   - **Readability features** (syllable counts, Flesch-Kincaid index, token statistics, linguistic patterns specific to Hungarian)
   - **CORAL Loss** (Correct Ordinal Regression via All-threshold Loss) to leverage ordinal nature of readability scale (1 < 2 < 3 < 4 < 5)

This fusion approach treats the prediction task as ordinal regression rather than nominal classification, which is more semantically appropriate for readability scoring.

### Training Methodology
- **Data Pipeline**: Fully automated 6-step pipeline
  1. Data acquisition & exploratory analysis (read-only)
  2. Cleansing, normalization, deduplication with stratified train/val/test split
  3. Baseline transformer training with hyperparameter tuning
  4. Advanced model (fusion) development with CORAL optimization
  5. Evaluation criteria definition and standard metrics computation
  6. Advanced evaluation: robustness testing (noise injection, truncation) and explainability analysis (attention-based token importance)

- **Optimization**: AdamW with linear warmup + cosine decay scheduler; mixed precision training with Accelerate
- **Data Split**: Stratified 60% train / 20% validation / 20% test
- **Evaluation Metrics**: MAE, RMSE, F1-score, confusion matrix, accuracy per class

### Results
- **Baseline Model**: Transformer-based classification achieving solid accuracy and F1 metrics
- **Advanced Model**: FusionModel with CORAL demonstrates ordinal-aware predictions with improved interpretability
- **Robustness**: Model resilience tested under noise injection and text truncation
- **Explainability**: Attention weights and token importance analysis provide transparency into model decisions

## Extra Credit Justification

This project targets the **Outstanding Level (+1 mark)** for the following reasons:

### 1. **Incremental Model Development & Innovation**
- Advanced FusionModel that combines transformer embeddings with domain-specific readability features
- CORAL loss for true ordinal regression (not just classification)
- Creative feature engineering for Hungarian legal text readability

### 2. **Advanced Evaluation**
- Robustness testing (noise injection, truncation) to assess model reliability
- Explainability analysis using attention-based token importance
- Comprehensive reporting: MAE, RMSE, F1, confusion matrices, per-class accuracy

### 3. **ML as a Service (Full Stack)**
- REST API (FastAPI) for backend model serving
- Streamlit GUI for interactive, user-friendly frontend
- End-to-end automation: data → model → API → GUI

### 4. **Production-Ready Containerization**
- Dockerized solution supporting GPU, multi-service orchestration
- Configurable service startup (training, API, frontend)
- Environment-driven, reproducible deployment

### 5. **Creative, Well-Developed Solutions**
- End-to-end automation from raw data to deployed service
- Modular, maintainable codebase with clear separation of concerns
- Domain-specific metrics and preprocessing for Hungarian legal texts
- Demonstrated exceptional performance and code quality

## Data Preparation

### Data Source Format
The project expects raw data in **Label Studio JSON format** with the following structure:
```json
[
  {
    "data": { "text": "Legal text here..." },
    "annotations": [{ "result": [{ "value": { "text": ["1"] }, ... }] }]
  },
  ...
]
```

### Automated Data Pipeline
The pipeline is **fully automated** through `src/run.sh` and includes:

1. **Step 01 – Data Acquisition** (`src/01_data_acquisition_and_analysis.py`)
   - Loads all JSON files from `/app/data` directory
   - Performs exploratory data analysis (EDA)
   - Generates statistics: label distribution, text length histograms, TF-IDF analysis
   - Output: `raw/raw_dataset.csv` + EDA visualizations
   - **Read-only**: No modification of raw data

2. **Step 02 – Cleansing & Preparation** (`src/02_data_cleansing_and_preparation.py`)
   - **Text normalization**: Lowercasing, Unicode NFC normalization, whitespace collapse
   - **Special character filtering**: Preserves legal symbols (—, …, €, etc.) while removing noise
   - **Deduplication**: On lowercased text to identify and remove duplicates
   - **Train/Val/Test Split**: Stratified split (60/20/20) on label distribution
   - Output: `processed/train.csv`, `processed/val.csv`, `processed/test.csv` + text statistics

### Data Access
- **Raw data location**: Mount data to `/app/data` inside container
- **Processed data location**: Automatically saved to `/app/output/processed/`
- **Logging**: Full data processing steps logged to `training_log.txt`

## Logging

The training process captures comprehensive logging required for grading. All output is automatically teed to `training_log.txt` in `OUTPUT_DIR`.

### Log Contents
1. **Configuration**: Batch size, learning rate, epochs, warmup steps, model name, optimizer
2. **Data Processing**: Confirmation of raw data loading, item counts, deduplication statistics, split sizes
3. **Model Architecture**: Model type, parameter counts (trainable/non-trainable), layer details
4. **Training Progress**: Per-epoch loss (training), accuracy, F1-score
5. **Validation**: Per-epoch validation metrics (MAE, RMSE, F1)
6. **Final Evaluation**: Test set MAE, RMSE, F1-score, accuracy per class, confusion matrix
7. **Advanced Evaluation**: Robustness test results (accuracy under noise/truncation), token importance analysis

### Docker Log Capture
```powershell
docker run --rm --gpus all `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > log/run.log 2>&1
```

## Docker Instructions

### Build

Run the following command in the root directory to build the Docker image:

```bash
docker build -t deeplearning_project-legal_text_decoder:1.0 .
```

### Run

The container supports multiple execution modes via environment variables. All modes output logs to `training_log.txt` in the mounted output volume.

#### Mode 1: Training Only (Basic)
Runs complete pipeline: data preparation, baseline, advanced model, and evaluation.

**Windows PowerShell:**
```powershell
docker run --rm --gpus all `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > log/run.log 2>&1
```

**Linux/macOS:**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/attach_folders/data:/app/data" \
  -v "$(pwd)/attach_folders/output:/app/output" \
  deeplearning_project-legal_text_decoder:1.0 \
  > log/run.log 2>&1
```

#### Mode 2: Training + API Service
Trains the model and starts FastAPI backend on port 8000.

**Windows PowerShell:**
```powershell
docker run --rm --gpus all `
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
  -p 8000:8000 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > log/run.log 2>&1
```

#### Mode 3: Full Stack (Training + API + Frontend)
Runs complete pipeline, FastAPI backend, and Streamlit GUI (ports 8000 & 8501).

**Windows PowerShell:**
```powershell
docker run --rm --gpus all `
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
  -e START_FRONTEND_SERVICE=1 -e API_URL=http://localhost:8000 -e FRONTEND_HOST=0.0.0.0 -e FRONTEND_PORT=8501 `
  -p 8000:8000 -p 8501:8501 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > log/run.log 2>&1
```

**Note**: Replace volume paths with `$(pwd)/attach_folders/...` on Linux/macOS.

### Key Mount Points
- **`/app/data`**: Raw data directory (JSON files in Label Studio format)
- **`/app/output`**: Output directory for models, reports, processed data, and logs

### Access Services
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Streamlit Frontend**: `http://localhost:8501`
- **Logs**: Check `output/training_log.txt` inside mounted volume

## File Structure

The repository is organized as follows:

### Source Code (`src/`)
- **`01_data_acquisition_and_analysis.py`**: Load JSON data, generate EDA statistics and visualizations
- **`02_data_cleansing_and_preparation.py`**: Text normalization, deduplication, stratified split
- **`03_baseline_model.py`**: Train and evaluate baseline HuBERT transformer
- **`04_incremental_model_development.py`**: Advanced FusionModel with CORAL loss, readability features
- **`05_defining_evaluation_criteria.py`**: Standard evaluation pipeline (MAE, RMSE, F1, confusion matrix)
- **`06_advanced_evaluation.py`**: Robustness testing and explainability analysis (combined)
- **`07_start_api_service.py`**: FastAPI service startup script
- **`08_start_frontend_service.py`**: Streamlit GUI startup script
- **`run.sh`**: Main orchestration script executing full pipeline
- **`api/app.py`**: REST API endpoints and model inference logic
- **`frontend/app.py`**: Streamlit GUI for interactive inference

### Notebooks (`notebook/`)
- **`teszteles.ipynb`**: Testing and experimental notebook

### Configuration
- **`Dockerfile`**: Docker image definition with CUDA 12.9 + PyTorch 2.8
- **`requirements.txt`**: Python dependencies with pinned versions
- **`LICENSE`**: Project license

### Output (`output/` - created at runtime)
- **`raw/`**: Raw data CSV and EDA charts
- **`processed/`**: Cleaned train/val/test splits
- **`models/`**: Saved baseline and transformer models
- **`reports/`**: Evaluation reports and metrics visualizations
- **`training_log.txt`**: Comprehensive training log

## Dependencies

All dependencies are listed in `requirements.txt` with pinned versions for reproducibility:

```
numpy<2,>=1.23.2
pandas>=2.0.0
scikit-learn>=1.5.0
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=2.6.0
fastapi>=0.110.0
uvicorn>=0.29.0
streamlit>=1.32.0,<2.0.0
textstat>=0.7.3
matplotlib>=3.8.0
seaborn>=0.12.0
plotly>=5.20.0
```