# Legal Text Decoder - Gyors Referencia Kártya

## 🚀 Leggyakoribb Parancsok

### 1️⃣ Image Build
```powershell
docker build -t deeplearning_project-legal_text_decoder:1.0 .
```

### 2️⃣ Teljes Pipeline Futtatás
```powershell
docker run --rm --gpus all `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > training_log.txt 2>&1
```

### 3️⃣ Csak Baseline (Gyorsabb, CPU)
Átnevezés: `04_train_transformer.py` → `04_train_transformer.py.bak`
```powershell
docker run --rm `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > training_log.txt 2>&1
```

### 4️⃣ Konfiguráció Változtatás
```powershell
docker run --rm --gpus all `
  -e EPOCHS=5 `
  -e BATCH_SIZE=16 `
  -e LEARNING_RATE=3e-5 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0
```

---

## 🔧 Környezeti Változók Cheat Sheet

### Data & Output
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `DATA_DIR` | `/app/data` | Input adatok helye |
| `OUTPUT_DIR` | `/app/output` | Kimenetek helye |

### Baseline Model (TF-IDF + LogReg)
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `TFIDF_MAX_FEATURES` | `20000` | Max TF-IDF feature-ök |
| `TFIDF_NGRAM_MAX` | `2` | N-gram felső határ |
| `LR_C` | `1.0` | Regularizációs paraméter |

### Transformer Model
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `TRANSFORMER_MODEL` | `SZTAKI-HLT/hubert-base-cc` | Modell név |
| `BATCH_SIZE` | `8` | Batch méret |
| `EPOCHS` | `3` | Epochok száma |
| `LEARNING_RATE` | `2e-5` | Tanulási ráta |
| `MAX_LENGTH` | `512` | Max token hossz |

### Feature Engineering
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `ENABLE_EMBEDDINGS` | `false` | Sentence-BERT be/ki |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding modell |

---

## 📂 Output Struktúra

```
output/
├── processed/          # CSV adatok (train/val/test)
├── features/           # Statisztikák, hisztogramok, embeddings
├── models/             # baseline_model.pkl, transformer_model/
├── reports/            # Metrikák JSON-ben, confusion matrix-ok
├── evaluation/         # Test eredmények
├── robustness/         # Robusztussági tesztek
└── explainability/     # Feature importance, magyarázatok
```

---

## 🐛 Gyors Hibaelhárítás

### GPU nem működik
```powershell
# Ellenőrizd:
nvidia-smi

# Ha nem működik, telepítsd:
# NVIDIA Container Toolkit
```

### Out of Memory
```powershell
# Csökkentsd a batch size-t:
docker run --rm --gpus all -e BATCH_SIZE=4 ...
```

### Lassú futás
```powershell
# Csak baseline (gyorsabb):
# Nevezd át vagy töröld: 04_train_transformer.py
```

### Stratified Split Hiba
```
# Legalább 3-5 példa kell osztályonként
# Ellenőrizd az adatokat!
```

---

## 📊 Pipeline Lépések

1. **01_data_processing.py** → CSV generálás (train/val/test)
2. **02_feature_engineering.py** → Statisztikák + embeddings
3. **03_train_baseline.py** → TF-IDF + LogReg
4. **04_train_transformer.py** → HuBERT finomhangolás (GPU!)
5. **05_evaluation.py** → Test értékelés
6. **06_robustness_tests.py** → Zaj/csonkolás tesztek
7. **07_explainability.py** → Feature importance, magyarázatok

---

## 📈 Benchmark Idők

| Lépés | CPU | GPU (RTX 3080) |
|-------|-----|----------------|
| Data Processing | ~30s | ~30s |
| Feature Engineering | ~1min | ~1min |
| Baseline Training | ~3min | ~3min |
| Transformer Training | ~6h+ | ~30-45min |
| Evaluation | ~30s | ~10s |
| Robustness Tests | ~2min | ~1min |
| Explainability | ~1min | ~30s |
| **TOTAL** | ~6h+ | **~40-55min** |

---

## 🔑 Kulcs Fájlok

| Fájl | Cél |
|------|-----|
| `Dockerfile` | Docker környezet |
| `requirements.txt` | Python csomagok |
| `src/run.sh` | Pipeline orchestration |
| `src/01-07_*.py` | Pipeline lépések |
| `README.md` | Használati útmutató |
| `ARCHITECTURE.md` | Technikai dokumentáció |
| `SUBMISSION.md` | Beadási dokumentáció |

---

## 💡 Pro Tippek

### 1. Gyors Iteráció
Kommenteld ki a hosszú lépéseket a `run.sh`-ban fejlesztés alatt:
```bash
# python 04_train_transformer.py  # Kihagyás
```

### 2. Memory Optimization
Ha kevés a memória:
```powershell
-e BATCH_SIZE=4 -e MAX_LENGTH=256
```

### 3. Quick Test
Csak 1 epoch teszteléshez:
```powershell
-e EPOCHS=1
```

### 4. Log Monitoring
Valós idejű log követés:
```powershell
docker logs -f <container_id>
```

### 5. Disk Space
A transformer model nagy (~500MB). Figyelj a disk space-re!

---

## 📞 Gyakori Kérdések

**Q: Mennyi időbe telik a teljes futás?**  
A: GPU-val ~45-60 perc, CPU-val 6+ óra (transformer miatt).

**Q: Kell-e internet?**  
A: Csak az első futásnál (model letöltéshez). Utána offline is megy.

**Q: Mekkora GPU kell?**  
A: Min. 8GB VRAM (batch_size=8), ajánlott 16GB+.

**Q: Működik CPU-n?**  
A: Igen, de a transformer nagyon lassú. Baseline gyors.

**Q: Hány adatra van szükség?**  
A: Min. ~100-200 példa, ajánlott 1000+.

**Q: Támogat más nyelveket?**  
A: Igen! Változtasd meg a `TRANSFORMER_MODEL` változót.

---

## 📚 Hasznos Linkek

- [PyTorch Docs](https://pytorch.org/docs/)
- [Hugging Face Models](https://huggingface.co/models)
- [Docker GPU Setup](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [scikit-learn](https://scikit-learn.org/stable/)

---

**Készült:** 2025. November  
**Quick Reference:** v1.0  
**Projekt:** Legal Text Decoder
