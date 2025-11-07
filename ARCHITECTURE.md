# Legal Text Decoder - Projekt Architektúra és Részletes Dokumentáció

## 📋 Projekt Áttekintés

A **Legal Text Decoder** egy mélytanulás alapú NLP rendszer, amely automatikusan értékeli jogi szövegek (ÁSZF, ÁFF) érthetőségét egy 1-5 skálán. A projekt Docker konténerben fut, NVIDIA GPU támogatással.

## 🏗️ Architektúra

### Komponensek

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Data Processing (01_data_processing.py)            │ │
│  │     - JSON betöltés és validálás                       │ │
│  │     - Szöveg tisztítás (Unicode, whitespace)           │ │
│  │     - Stratifikált train/val/test split                │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  2. Feature Engineering (02_feature_engineering.py)    │ │
│  │     - Szövegstatisztikák (word count, avg word len)    │ │
│  │     - Opcionális Sentence-BERT embeddings              │ │
│  │     - Exploratív adatvizualizáció                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  3. Baseline Model (03_train_baseline.py)             │ │
│  │     - TF-IDF vektorizáció (max 20k features)           │ │
│  │     - Logistic Regression klasszifikáció              │ │
│  │     - Sklearn Pipeline                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  4. Transformer Model (04_train_transformer.py)        │ │
│  │     - HuBERT finomhangolás (magyar BERT)               │ │
│  │     - PyTorch + Transformers library                   │ │
│  │     - GPU akceleráció                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  5. Evaluation (05_evaluation.py)                      │ │
│  │     - Test set értékelés                               │ │
│  │     - Classification report, confusion matrix          │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  6. Robustness Tests (06_robustness_tests.py)          │ │
│  │     - Zaj-tűrés tesztelés (5%, 10%, 20% noise)         │ │
│  │     - Csonkolás tesztelés (75%, 50%, 25%)              │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  7. Explainability (07_explainability.py)              │ │
│  │     - Feature importance elemzés                       │ │
│  │     - Predikció magyarázatok                           │ │
│  │     - Hibaelemzés                                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Projekt Struktúra

```
DeepLearning_Project-Legal_Text_Decoder/
├── Dockerfile                      # Docker környezet definíció
├── requirements.txt                # Python függőségek
├── README.md                       # Projekt dokumentáció
├── ARCHITECTURE.md                 # Ez a fájl
├── LICENSE                         # Licenc
├── .gitignore                      # Git ignore fájl
├── RUNNING_DOCKERFILE.txt          # Docker futtatási utasítások
├── training_log.txt               # Pipeline futás logja (generált)
│
├── notebook/                       # Jupyter notebookok kísérletezéshez
│   └── teszteles.ipynb
│
└── src/                            # Forráskód
    ├── run.sh                      # Pipeline futtatási script
    ├── 01_data_processing.py       # Adat előkészítés
    ├── 02_feature_engineering.py   # Feature extraction
    ├── 03_train_baseline.py        # Baseline modell
    ├── 04_train_transformer.py     # Transformer modell
    ├── 05_evaluation.py            # Kiértékelés
    ├── 06_robustness_tests.py      # Robusztussági tesztek
    └── 07_explainability.py        # Magyarázhatóság
```

## 🔧 Technológiai Stack

### Core Technologies
- **Python 3.11+**: Fő programozási nyelv
- **Docker**: Konténerizáció és reprodukálható környezet
- **CUDA 12.9 + cuDNN 9**: GPU támogatás

### Machine Learning & Deep Learning
- **PyTorch 2.8.0**: Deep learning framework
- **Transformers 4.40.0**: Hugging Face transformer modellek
- **scikit-learn 1.5.2**: Baseline ML modellek
- **sentence-transformers 2.6.1**: Sentence embeddings

### Data Processing
- **pandas 2.3.3**: Adatkezelés
- **numpy 2.3.3**: Numerikus műveletek
- **tqdm 4.66.5**: Progress bar-ok

### Visualization
- **matplotlib 3.10.7**: Ábrák és vizualizációk

## 🎯 Részletes Pipeline Leírás

### 1. Data Processing (01_data_processing.py)

**Bemenet:** JSON fájl(ok) az `/app/data` mappában

**Kimenet:** `train.csv`, `val.csv`, `test.csv` az `/app/output/processed/` mappában

**Fő funkciók:**
- `load_json_data()`: Egyetlen JSON fájl betöltése
- `load_json_items()`: Több JSON fájl vagy mappa feldolgozása
- `clean_text()`: Unicode normalizálás, whitespace tisztítás, speciális karakterek kezelése
- `stratified_split()`: 60% train, 20% val, 20% test stratifikált osztás

**Adatséma:**
```json
{
  "data": { "text": "A szöveg bekezdése..." },
  "annotations": [{
    "result": [{
      "value": { "choices": ["Könnyen érthető"] }
    }]
  }]
}
```

**Adattisztítás:**
- NFC Unicode normalizálás (magyar ékezetek megőrzése)
- Többszörös whitespace eltávolítása
- Speciális karakterek szűrése, jogi írásjelek megtartása
- Üres szövegek és hiányzó labelek kiszűrése

### 2. Feature Engineering (02_feature_engineering.py)

**Bemenet:** Processed CSV-k

**Kimenet:** Kiegészített CSV-k + vizualizációk + opcionális embeddings

**Szövegstatisztikák:**
- `word_count`: Szavak száma
- `avg_word_len`: Átlagos szóhossz

**Opcionális Embeddings:**
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Formátum: NumPy tömbök (.npy)
- Metadata: JSON fájl az embeddings helyével

**Vizualizációk:**
- Word count hisztogram (train set)
- Average word length hisztogram (train set)

**Környezeti változók:**
- `ENABLE_EMBEDDINGS=true`: Embeddings generálás bekapcsolása
- `EMBEDDING_MODEL`: Sentence-BERT modell neve

### 3. Baseline Model (03_train_baseline.py)

**Architektúra:**
```
Text → TF-IDF Vectorizer → Logistic Regression → Prediction
```

**TF-IDF konfiguráció:**
- N-gram range: (1, 2) - unigram és bigram
- Max features: 20,000 (konfigurálható)
- Tokenization: alapértelmezett
- Stopwords: nincs (jogi szöveg specifikus szavak fontosak)

**Logistic Regression:**
- Multi-class: One-vs-Rest
- Max iterations: 1000
- Regularization: L2 (C=1.0, konfigurálható)
- Solver: lbfgs

**Kimenetek:**
- `baseline_model.pkl`: Sklearn Pipeline
- `baseline_val_report.json`: Validációs metrikák
- `baseline_test_report.json`: Test metrikák
- `baseline_test_confusion_matrix.png`: Konfúziós mátrix

**Környezeti változók:**
- `TFIDF_MAX_FEATURES=20000`: TF-IDF feature-ök maximális száma
- `TFIDF_NGRAM_MAX=2`: N-gram felső határ
- `LR_C=1.0`: Regularizációs paraméter

### 4. Transformer Model (04_train_transformer.py)

**Alapértelmezett modell:** `SZTAKI-HLT/hubert-base-cc` (magyar BERT)

**Architektúra:**
```
Text → Tokenizer → HuBERT Encoder → Classification Head → Softmax
```

**Tokenizáció:**
- Max length: 512 token
- Padding: max_length
- Truncation: True
- Special tokens: [CLS], [SEP]

**Fine-tuning stratégia:**
- Optimizer: AdamW
- Learning rate: 2e-5 (with warmup)
- Warmup: 10% of total steps
- Gradient clipping: 1.0
- Batch size: 8 (konfigurálható)
- Epochs: 3 (konfigurálható)

**Training loop:**
1. Forward pass
2. Loss calculation (Cross-Entropy)
3. Backward pass
4. Gradient clipping
5. Optimizer step
6. Scheduler step
7. Metrics logging

**Kimenetek:**
- `transformer_model/`: Teljes modell (config, weights, tokenizer)
- `label_mapping.json`: Label → ID mapping
- `transformer_training_history.png`: Tanítási görbék
- `transformer_test_report.json`: Test metrikák

**Környezeti változók:**
- `TRANSFORMER_MODEL`: Használandó modell neve
- `BATCH_SIZE=8`: Batch méret
- `EPOCHS=3`: Epochok száma
- `LEARNING_RATE=2e-5`: Tanulási ráta
- `MAX_LENGTH=512`: Max token hossz

**GPU követelmények:**
- Minimum: 8GB VRAM (batch size 8-hoz)
- Ajánlott: 16GB+ VRAM (nagyobb batch size-hoz)

### 5. Evaluation (05_evaluation.py)

**Metrikák:**
- Accuracy (összes osztályra)
- Precision, Recall, F1-score (osztályonként)
- Support (példák száma osztályonként)
- Macro avg (egyenlő súlyozás)
- Weighted avg (mintaszám szerinti súlyozás)

**Vizualizációk:**
- Confusion matrix heatmap
- Per-class performance

**Kimenetek:**
- `baseline_test_report.json`: Részletes metrikák
- `baseline_test_confusion_matrix.png`: Konfúziós mátrix

### 6. Robustness Tests (06_robustness_tests.py)

**Tesztelt perturbációk:**

1. **Karakter-szintű zaj:**
   - 5% zaj: Véletlenszerű karaktermódosítás
   - 10% zaj: Közepesen zajos szöveg
   - 20% zaj: Erősen zajos szöveg

2. **Szöveg csonkolás:**
   - 75%: Enyhe információvesztés
   - 50%: Fele hosszúság
   - 25%: Csak az első negyedév

**Zaj műveletek:**
- Delete: Karakter törlése
- Duplicate: Karakter duplikálása
- Space: Karakter lecserélése space-re

**Kimenetek:**
- `robustness_results.json`: Minden teszt részletes eredménye
- `robustness_comparison.png`: Accuracy összehasonlítás

**Értékelés:**
- Baseline accuracy (eredeti szöveg)
- Degradáció mérése (accuracy csökkenés)
- Robusztusság score

### 7. Explainability (07_explainability.py)

**Feature Importance:**
- Top 20 legfontosabb szó/n-gram osztályonként
- Logistic Regression coefficients alapján
- Pozitív és negatív súlyok

**Prediction Explanations:**
- Top 10 test példa
- True vs. Predicted label
- Top 3 predikció probability-vel
- Helyesség jelölés

**Misclassification Analysis:**
- Összes hibás predikció
- Hibapárok gyakorisága (true → predicted)
- Top 10 legtöbb hibapár
- Példa hibás predikciók

**Kimenetek:**
- `feature_importance.json`: Feature súlyok
- `top_features_per_class.png`: Feature importance plot-ok
- `prediction_explanations.json`: Predikció magyarázatok
- `misclassification_analysis.json`: Hibaelemzés

## 🐳 Docker Konfiguráció

### Base Image
```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
```

**Előnyök:**
- PyTorch és CUDA előre telepítve
- Optimalizált GPU használat
- Kisebb image méret (runtime vs. devel)

### Volume Mounting
```bash
-v "C:\...\data:/app/data"          # Input data
-v "C:\...\output:/app/output"      # Results
```

### GPU Access
```bash
--gpus all                           # Minden GPU elérése
```

### Környezeti változók átadása
```bash
docker run -e EPOCHS=5 -e BATCH_SIZE=16 ...
```

## 📊 Kimenet Struktúra

```
output/
├── processed/
│   ├── train.csv               # Training set (60%)
│   ├── val.csv                 # Validation set (20%)
│   └── test.csv                # Test set (20%)
│
├── features/
│   ├── train_word_count_hist.png
│   ├── train_avg_word_len_hist.png
│   ├── embeddings_train.npy    (opcionális)
│   ├── embeddings_val.npy      (opcionális)
│   ├── embeddings_test.npy     (opcionális)
│   └── embeddings_meta.json    (opcionális)
│
├── models/
│   ├── baseline_model.pkl
│   ├── label_mapping.json
│   └── transformer_model/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
│
├── reports/
│   ├── baseline_val_report.json
│   ├── baseline_test_report.json
│   ├── baseline_test_confusion_matrix.png
│   ├── transformer_test_report.json
│   └── transformer_training_history.png
│
├── evaluation/
│   ├── baseline_test_report.json
│   └── baseline_test_confusion_matrix.png
│
├── robustness/
│   ├── robustness_results.json
│   └── robustness_comparison.png
│
└── explainability/
    ├── feature_importance.json
    ├── top_features_per_class.png
    ├── prediction_explanations.json
    └── misclassification_analysis.json
```

## 🚀 Futtatási Útmutató

### 1. Image Build
```powershell
docker build -t deeplearning_project-legal_text_decoder:1.0 .
```

### 2. Pipeline Futtatás
```powershell
docker run --rm --gpus all `
  -v "C:\path\to\data:/app/data" `
  -v "C:\path\to\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > training_log.txt 2>&1
```

### 3. Egyedi Konfiguráció
```powershell
docker run --rm --gpus all `
  -e EPOCHS=5 `
  -e BATCH_SIZE=16 `
  -e ENABLE_EMBEDDINGS=true `
  -v "C:\path\to\data:/app/data" `
  -v "C:\path\to\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0
```

### 4. Csak Baseline (gyorsabb)
Töröld vagy nevezd át a `04_train_transformer.py` fájlt futtatás előtt.

## 🔍 Hibaelhárítás

### GPU nem elérhető
**Probléma:** `CUDA not available`

**Megoldás:**
1. Ellenőrizd: `nvidia-smi` parancs működik-e
2. Docker Desktop GPU támogatás bekapcsolva
3. NVIDIA Container Toolkit telepítve

### Memória túlcsordulás (OOM)
**Probléma:** `RuntimeError: CUDA out of memory`

**Megoldás:**
- Csökkentsd a batch size-t: `-e BATCH_SIZE=4`
- Csökkentsd a max length-et: `-e MAX_LENGTH=256`
- Használj kisebb modellt: `-e TRANSFORMER_MODEL=bert-base-multilingual-cased`

### Stratifikált split hiba
**Probléma:** `ValueError: The least populated class has only 1 member`

**Megoldás:**
- Több adat szükséges
- Minimum 3-5 példa osztályonként
- Ellenőrizd az adatokat: label eloszlás

### Lassú futás
**Probléma:** Transformer tanítás nagyon lassú

**Megoldás:**
- GPU használat ellenőrzése
- Kisebb epoch szám: `-e EPOCHS=2`
- Csak baseline futtatása (töröld a 04-es scriptet)

## 📈 Teljesítmény Benchmark-ok

### Baseline Model (TF-IDF + LogReg)
- **Training idő:** ~2-5 perc (CPU)
- **Prediction idő:** ~1 ms/document
- **Memory:** ~500 MB
- **Tipikus accuracy:** 60-75%

### Transformer Model (HuBERT)
- **Training idő:** ~30-60 perc (GPU, 3 epoch)
- **Prediction idő:** ~50 ms/document (GPU)
- **Memory:** ~2-4 GB (training), ~1 GB (inference)
- **Tipikus accuracy:** 70-85%

## 🔐 Biztonság és Adatvédelem

- **Adatok:** Csak lokális Docker volume-okban
- **Modellek:** Offline inference lehetséges
- **Nincsenek külső API hívások** (embeddings kivételével, ha be van kapcsolva)

## 📝 Best Practices

1. **Version Control:** 
   - Commitolj minden változtatást
   - Ne commitolj data/ és output/ mappákat
   - Használd a .gitignore-t

2. **Reproducibility:**
   - Rögzített random seed-ek a split-hez (42)
   - Docker image verziókezelés
   - Requirements.txt pontos verziószámokkal

3. **Monitoring:**
   - training_log.txt folyamatos ellenőrzése
   - GPU utilization monitoring (nvidia-smi)
   - Disk space monitoring (modellek nagy mérete)

4. **Optimization:**
   - Mixed precision training (fp16) transformerhez
   - Gradient accumulation kis batch size-nál
   - Model distillation nagyobb modellekből

## 📚 További Források

- [PyTorch dokumentáció](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [scikit-learn dokumentáció](https://scikit-learn.org/)
- [Docker dokumentáció](https://docs.docker.com/)

## 🤝 Közreműködés

A projekt a BME Deep Learning kurzus keretében készült. Minden javítás és új feature szívesen látott!

---

**Készült:** 2025. November  
**Verzió:** 1.0  
**Licenc:** Lásd LICENSE fájl
