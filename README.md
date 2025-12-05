
# Deep Learning Class (VITMMA19) - Legal Text Decoder

**Magyar jogi szövegek (ÁSZF/ÁFF) érthetőségének automatikus értékelése (1-5 skála) modern NLP-vel.**

**Főbb jellemzők:**
- Transformer (HuBERT) + olvashatósági feature fusion (FusionModel)
- Ordinal label mapping, CORAL loss támogatás
- Robusztus értékelés (zaj, csonkítás), attention-alapú magyarázhatóság
- REST API (FastAPI) + Web GUI (Streamlit)
- **Minden fájlírás UTF-8 kódolással történik** (magyar karakterek támogatása)
- Docker + GPU támogatás | Cross-platform

## 📚 Tartalomjegyzék

- Gyors Indítás
- Követelmény-Fájl Megfeleltetés
- Pipeline Lépések
- Statisztikai Elemzések
- Adatformátum
- Környezeti Változók
- Kimenetek
- ML Service - API + GUI
- Hibaelhárítás

---


## 🚀 Gyors Indítás

### Összefoglaló táblázat

| Mit szeretnél? | Gyors módszer | Docker Compose | Mit kapsz? |
|---------------|---------------|----------------|------------|
| **Csak training** | `\.\docker-run.ps1` **vagy** egyparancsos `docker run` | nem szükséges | Pipeline (01-07) |
| **Training + API** | `\.\docker-run-with-api.ps1` **vagy** egyparancsos `docker run` | opcionális | Pipeline + REST API (8000) |
| **Training + API + GUI** ⭐ | `\.\docker-run-full-stack.ps1` **vagy** egyparancsos `docker run` | opcionális | Pipeline + API (8000) + Frontend (8501) |
| **Csak API (kész modellel)** | `docker run` | opcionális | REST API (8000) |
| **API + Frontend (kész modellel)** | `docker run` | opcionális | API (8000) + GUI (8501) |

### 1. Csak Pipeline (alapértelmezett)

**Automatikus platform detektálás:**
```bash
# Windows PowerShell
.\docker-run.ps1

# Linux/macOS/Git Bash
bash docker-run.sh
```

**Manuális Docker futtatás (compose nélkül):**
```powershell
# 1. Build
docker build -t deeplearning_project-legal_text_decoder:1.0 .

# 2. Csak training (Windows PowerShell)
docker run --rm --gpus all `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 `
  > training_log.txt 2>&1

# 2b. Csak training (Linux/macOS)
docker run --rm --gpus all \
  -v "$(pwd)/attach_folders/data:/app/data" \
  -v "$(pwd)/attach_folders/output:/app/output" \
  deeplearning_project-legal_text_decoder:1.0 \
  > training_log.txt 2>&1
```

### 2. Pipeline + API indítás (egy lépésben) ⭐

**Automatikus script (legegyszerűbb):**
```bash
# Windows PowerShell
.\docker-run-with-api.ps1

# Linux/macOS/Git Bash
bash docker-run-with-api.sh
```

**Docker Compose módszer:**
```bash
docker-compose up training-with-api
```

**Manuális futtatás (START_API_SERVICE=1, compose nélkül):**
```powershell
# Windows PowerShell – training + API
docker run --rm --gpus all `
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
  -p 8000:8000 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 `
  > training_log.txt 2>&1

# Linux/macOS
docker run --rm --gpus all \
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 \
  -p 8000:8000 \
  -v "$(pwd)/attach_folders/data:/app/data" \
  -v "$(pwd)/attach_folders/output:/app/output" \
  deeplearning_project-legal_text_decoder:1.0 \
  > training_log.txt 2>&1
```

Ezután az API elérhető: http://localhost:8000

### 3. Pipeline + API + Frontend (teljes stack) ⭐ ÚJ

**Automatikus script (legegyszerűbb):**
```bash
# Windows PowerShell
.\docker-run-full-stack.ps1

# Linux/macOS/Git Bash
bash docker-run-full-stack.sh
```

**Docker Compose módszer:** (opcionális, nem szükséges)
```bash
docker-compose up training-full-stack
```

**Manuális futtatás (compose nélkül, training + API + Frontend):**
```powershell
# Windows PowerShell
docker run --rm --gpus all `
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
  -e START_FRONTEND_SERVICE=1 -e API_URL=http://localhost:8000 -e FRONTEND_HOST=0.0.0.0 -e FRONTEND_PORT=8501 `
  -p 8000:8000 -p 8501:8501 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 `
  > training_log.txt 2>&1

# Linux/macOS
docker run --rm --gpus all \
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 \
  -e START_FRONTEND_SERVICE=1 -e API_URL=http://localhost:8000 -e FRONTEND_HOST=0.0.0.0 -e FRONTEND_PORT=8501 \
  -p 8000:8000 -p 8501:8501 \
  -v "$(pwd)/attach_folders/data:/app/data" \
  -v "$(pwd)/attach_folders/output:/app/output" \
  deeplearning_project-legal_text_decoder:1.0 \
  > training_log.txt 2>&1
```

Ezután elérhető:
- **API**: http://localhost:8000
- **Frontend GUI**: http://localhost:8501

### 4. Csak API (már kész modellekkel)

```bash
# Docker Compose
docker-compose up api

# vagy manuálisan
cd src
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

> **A pipeline minden fájlírása UTF-8 kódolással történik.**
> A futás a teljes 01→05 + advanced (robustness + explainability) pipeline-t végigviszi. A baseline (03) opcionális.

---


## 🎯 Követelmény-Fájl Megfeleltetés (2024)

| # | Outstanding Level Követelmény | Implementáció | Fájl |
|---|-------------------------------|---------------|------|
| 1 | **Containerization** | Docker + GPU támogatás | `Dockerfile` |
| 2 | **Data acquisition and analysis** | JSON parser, RAW EDA, advanced statistics | `01_data_acquisition_and_analysis.py` |
| 3 | **Data cleansing and preparation** | Text cleaning, deduplication, stratified split | `02_data_cleansing_and_preparation.py` |
| 4 | **Defining evaluation criteria** | Transformer (HuBERT) batch inference, metrics, confusion matrix | `05_defining_evaluation_criteria.py` |
| 5 | **Baseline model (opcionális)** | TF-IDF + LogisticRegression | `03_baseline_model.py` |
| 6 | **Incremental model development** | Transformer (HuBERT) fine-tuning, feature fusion, ordinal mapping, CORAL loss | `04_incremental_model_development.py` |
| 7 | **Advanced evaluation** | Transformer-based Robustness + Explainability | `advanced_evaluation.py` (robustness + explainability egyben) |
| 8 | **ML as a service** | REST API + Web GUI + Pipeline Integration | `src/api/app.py` <br> `src/frontend/app.py` <br> `08_start_api_service.py` <br> `09_start_frontend_service.py` |

---


## 📋 Pipeline Lépések (2024)

Névkonvenció a kimenetekre: minden mérési/ábra/riport fájl név elején lépés-prefix szerepel.
Minta: `{lépés}-{rövid_név}_{típus}_{split}.{ext}`
Példák: `01-acquisition_raw_eda_statistics.txt`, `03-baseline_test_confusion_matrix.png`, `advanced/robustness/robustness_results.json`.

### 1. 01_data_acquisition_and_analysis.py
**Cél:** Nyers adatok betöltése és átfogó feltáró elemzés (EDA)

**Funkciók:**
- JSON adatok betöltése (fájl vagy mappa)
- Label kinyerés az annotations mezőből
- **Duplikátumok kiszűrése** (text alapján)
- **Hiányzó vagy üres címkék/szövegek eltávolítása**
- **RAW EDA:**
  - Word count és átlagos szóhossz eloszlások
  - Label eloszlás vizualizáció
- **Advanced statisztikai elemzések:**
  - **Olvashatósági metrikák:** Flesch Reading Ease, Gunning Fog Index, SMOG Index
  - **Lexikai diverzitás:** Type-Token Ratio (TTR), Moving Average TTR (MATTR), Hapax legomena
  - **TF-IDF top szavak** címkénként
  - **Korrelációs mátrix:** szöveghossz, olvashatóság, címke, diverzitás kapcsolatai
  - **Box plot vizualizációk** minden metrikára címkénként

**Kimenetek:**
- `output/raw/raw_dataset.csv` — teljes nyers adathalmaz
- `output/raw/raw_dataset_eda_filtered.csv` — deduplikált, szűrt snapshot
- `output/raw/raw_dataset_eda_enhanced.csv` — összes metrikával bővített adathalmaz
- `output/raw/removed_duplicates.csv`, `output/raw/removed_missing_labels.csv`
- `output/reports/01-acquisition_raw_eda_statistics.txt`
- `output/reports/01-acquisition_raw_label_distribution.png`
- `output/reports/01-acquisition_correlation_matrix.png`
- `output/reports/01-acquisition_tfidf_top_words_by_label.csv`
- `output/reports/01-acquisition_*_by_label.png` — 6 db boxplot (olvashatóság + diverzitás)

### 2. 02_data_cleansing_and_preparation.py
**Cél:** Szövegtisztítás és train/val/test split

**Funkciók:**
- Betölti a szűrt EDA adatokat (`raw_dataset_eda_filtered.csv`)
- Unicode normalizálás, whitespace kezelés
- Speciális karakterek szűrése (magyar jogi szövegekre optimalizálva)
- Stratified split (60% train, 20% val, 20% test)
- Szövegstatisztikák hozzáadása (word_count, avg_word_len)
- Opcionális: Sentence-BERT embeddings

**Kimenetek:**
- `output/processed/train.csv`, `val.csv`, `test.csv`
- `output/reports/02-preparation_clean_word_count_hist.png`
- `output/reports/02-preparation_clean_avg_word_len_hist.png`


### 3. 03_baseline_model.py (opcionális)
**Cél:** Baseline szövegklasszifikáció (gyors, CPU-barát)

**Modell:** TF-IDF (max_features=20000, ngram_range=(1,2)) + LogisticRegression (C=1.0)

**Kimenetek:**
- `output/models/baseline_model.pkl`
- `output/reports/03-baseline_val_report.json`, `03-baseline_test_report.json`
- `output/reports/03-baseline_val_confusion_matrix.png`, `03-baseline_test_confusion_matrix.png`
- `output/reports/03-baseline_val_metrics_summary.png`, `03-baseline_test_metrics_summary.png`
  (Accuracy, Weighted F1, MAE, RMSE vizuális összefoglaló)



### 4. 04_incremental_model_development.py
**Cél:** Transformer (HuBERT) fine-tuning, olvashatósági feature fusion (FusionModel), ordinal label mapping, CORAL loss, legjobb checkpoint mentése

**Fő fejlesztések:**
- Readability feature fusion (MLP branch, standardized)
- Mean pooling (stabilabb, mint CLS)
- Sqrt-scaled osztálysúlyok, label smoothing
- Ordinal label mapping (1–5 skála)
- **CORAL ordinal regression loss támogatás** (opcionális, `USE_CORAL=1`)
- Early stopping, checkpoint mentés
- **Minden fájlírás UTF-8 kódolással**

**Kimenetek:**
- `output/models/best_transformer_model/` — csak a legjobb checkpoint
- `output/models/label_mapping.json` — label-idx mapping
- `output/reports/04-transformer_training_history.png`
- `output/reports/04-transformer_test_report.json` (Accuracy, Macro/Weighted F1, MAE, RMSE)
- `output/reports/04-transformer_test_confusion_matrix.png`, per-class metrikák

**Legutóbbi eredmények (baseline-hoz képest):**

| Metrika         | Baseline (03) | Incrementális (04) |
|-----------------|---------------|--------------------|
| Accuracy        | 0.4474        | 0.4459             |
| Weighted F1     | 0.4158        | 0.4378             |
| Macro F1        | ~0.31         | 0.3664             |
| MAE             | 0.7674        | 0.7615             |
| RMSE            | 1.1392        | 1.1261             |

**Előnyök:**
- Macro F1 jelentősen javult (alulreprezentált osztályok)
- Weighted F1 stabilan jobb
- MAE/RMSE kismértékben csökkent
- Tanulás stabilabb, nincs ugrás az epochok között
- **Magyar karakterek mindenhol helyesen jelennek meg (UTF-8 encoding)**

**CORAL loss (opcionális):**
- További MAE/RMSE csökkenés, Macro F1 javulás várható
- Aktiválás: `USE_CORAL=1` környezeti változóval

### 5. 05_defining_evaluation_criteria.py
**Cél:** Transformer batch inference, metrikák, confusion matrix

**Kimenetek:**
- `output/reports/05-evaluation_test_report.json`
- `output/reports/05-evaluation_test_confusion_matrix.png`


### 6. advanced_evaluation.py
**Cél:** Robusztusság (zaj, csonkítás) és attention-alapú magyarázhatóság egy scriptben, közös inferencia réteggel.

**Kimenetek:**
- `output/reports/advanced/robustness/robustness_results.json`
- `output/reports/advanced/robustness/robustness_accuracy.png`
- `output/reports/advanced/explainability/attention_importance.json`
- `output/reports/advanced/explainability/attention_summary.json`
- `output/reports/advanced/explainability/misclassification_analysis.json`
- `output/reports/advanced/explainability/confusion_pairs.png`


### 7. (deprecated) 06/07 advanced scripts
Az előző két külön script (`06_advanced_evaluation_robustness.py`, `07_advanced_evaluation_explainability.py`) helyett használd az `advanced_evaluation.py` fájlt.

### 8. 08_start_api_service.py (opcionális) ⭐

**Cél:** API szerver indítása a pipeline befejezése után (FastAPI + uvicorn)

**Aktiválás:**
- Környezeti változó: `START_API_SERVICE=1`
- Docker Compose: `docker-compose up training-with-api`

**Funkciók:**
- Automatikus modell ellenőrzés (transformer + baseline)
- Konfigurálandó host és port (`API_HOST`, `API_PORT`)
- REST API endpoint a modell predikciókhoz
- Graceful shutdown (Ctrl+C)

**Kimenetek:**
- Indított API szerver: `http://0.0.0.0:8000` (vagy konfigurált port)
- Logok a konzolban/log fájlban

### 9. 09_start_frontend_service.py (opcionális) ⭐ ÚJ

**Cél:** Streamlit frontend indítása a pipeline befejezése után

**Aktiválás:**
- Környezeti változó: `START_FRONTEND_SERVICE=1`
- Docker Compose: `docker-compose up training-full-stack`

**Funkciók:**
- Webes GUI modell teszteléshez
- Konfigurálandó host és port (`FRONTEND_HOST`, `FRONTEND_PORT`)
- Automatikus API kapcsolat ellenőrzés
- Interaktív szöveg értékelés

**Kimenetek:**
- Indított Frontend szerver: `http://0.0.0.0:8501` (vagy konfigurált port)
- Logok a konzolban/log fájlban

> A `src/run.sh` sorban futtatja az összes `src/*.py` fájlt (01→07, opcionálisan 08-09). Dockerben ez az alapértelmezett belépési pont. A baseline (03), API service (08) és Frontend service (09) opcionális; a fő pipeline a transformer modellt használja minden értékeléshez.

---

## 📊 Statisztikai Elemzések

### Olvashatósági Indexek
- **Flesch Reading Ease** (0-100): magasabb érték = könnyebben olvasható
- **Gunning Fog Index**: hány év oktatás szükséges a megértéshez
- **SMOG Index**: komplex szavak alapú olvashatósági mutató

### Lexikai Diverzitás
- **Type-Token Ratio (TTR)**: egyedi szavak / összes szó
- **Moving Average TTR (MATTR)**: csúszó ablakos TTR (robusztusabb)
- **Hapax Legomena**: egyszer előforduló szavak aránya

### TF-IDF Elemzés
Minden érthetőségi kategóriára (1-5) meghatározza a legjellemzőbb szavakat/kifejezéseket.

### Korrelációs Analízis
Feltárja a kapcsolatokat:
- Szöveghossz ↔ Olvashatóság
- Lexikai diverzitás ↔ Érthetőségi címke
- Különböző metrikák közötti összefüggések

---

## 📄 Adatformátum (JSON)

Elvárt minimális séma egy elemre:

```json
{
  "data": { "text": "A bekezdés szövege…" },
  "annotations": [
    {
      "result": [
        { "value": { "choices": ["Könnyen érthető"] } }
      ]
    }
  ]
}
```

Fontos: ha több annotáció/eredmény van, jelenleg az első elem első választása kerül felhasználásra.

---

## ⚙️ Környezeti Változók

**Adatkezelés:**
- `DATA_DIR` — bemeneti adat mappa (alap: `/app/data` Dockerben)
- `OUTPUT_DIR` — kimeneti mappa (alap: `/app/output`)

**API Service (08_start_api_service.py):**
- `START_API_SERVICE` — API indítás a pipeline végén (alap: `0`, bekapcsolás: `1` vagy `true`)
- `API_HOST` — API szerver host címe (alap: `0.0.0.0`)
- `API_PORT` — API szerver portja (alap: `8000`)

**Frontend Service (09_start_frontend_service.py) - ÚJ:**
- `START_FRONTEND_SERVICE` — Frontend indítás a pipeline végén (alap: `0`, bekapcsolás: `1` vagy `true`)
- `FRONTEND_HOST` — Frontend szerver host címe (alap: `0.0.0.0`)
- `FRONTEND_PORT` — Frontend szerver portja (alap: `8501`)
- `API_URL` — API endpoint címe a frontend számára (alap: `http://localhost:8000`)

**Baseline (TF-IDF + LogisticRegression):**
- `TFIDF_MAX_FEATURES` (alap: `20000`), `TFIDF_NGRAM_RANGE` (alap: `1,2`), `LOGREG_C` (alap: `1.0`)

**Transformer (HuBERT) fine-tuning — jelenlegi alapértelmezések:**
- `TRANSFORMER_MODEL` — modell neve (alap: `SZTAKI-HLT/hubert-base-cc`)
- `EPOCHS` — max epoch (alap: `10`, early stopping miatt nem feltétlen fut végig)
- `BATCH_SIZE` — batch méret (alap: `8`)
- `LEARNING_RATE` — tanulási ráta (alap: `2e-5`)
- `WEIGHT_DECAY` — L2 regularizáció (alap: `0.01`)
- `MAX_LENGTH` — token hossz (alap: `320`)
- `LABEL_SMOOTHING` — label smoothing (alap: `0.15`)
- `EARLY_STOPPING` — engedélyezés (alap: `1`)
- `EARLY_STOPPING_PATIENCE` — türelem (alap: `2`), monitor: `val_macro_f1`
- `SAVE_BEST_METRIC` — `val_macro_f1` vagy `val_loss` (alap: `val_macro_f1`)
- `USE_CLASS_WEIGHTS` — osztálysúlyozás (alap: `1`)
- `USE_FOCAL_LOSS` — Focal Loss kapcsoló (alap: `0`), `FOCAL_GAMMA` (alap: `2.0`)
- `GRAD_ACC_STEPS` — grad. akkumuláció (alap: `2`)
- `MIXED_PRECISION` — automatikus FP16 (CUDA) (alap: `1`)

---

## 📦 Kimenetek

Az összes mérési és vizuális kimenet lépés-prefixet kap az egyszerű visszakövethetőségért.

### `output/raw/`
- `raw_dataset.csv`, `raw_dataset_eda_filtered.csv`, `raw_dataset_eda_enhanced.csv`
- `removed_duplicates.csv`, `removed_missing_labels.csv`

### `output/reports/` (EDA és cleaning)
- `01-acquisition_raw_eda_statistics.txt`, `01-acquisition_raw_label_distribution.png`
- `01-acquisition_correlation_matrix.png`, `01-acquisition_tfidf_top_words_by_label.csv`
- `01-acquisition_*_by_label.png` (6 db)
- `02-preparation_clean_word_count_hist.png`, `02-preparation_clean_avg_word_len_hist.png`

### `output/processed/`
- `train.csv` (~60%), `val.csv` (~20%), `test.csv` (~20%) — oszlopok: `text`, `label`, `word_count`, `avg_word_len`

### `output/models/`
- `baseline_model.pkl` — TF-IDF + LogisticRegression *(opcionális)*
- `best_transformer_model/` — HuBERT checkpoint + tokenizer (csak a legjobb)
- `label_mapping.json` — label-idx mapping

### `output/reports/`
- `03-baseline_val_report.json`, `03-baseline_test_report.json`
- `03-baseline_val_confusion_matrix.png`, `03-baseline_test_confusion_matrix.png`
- `03-baseline_val_metrics_summary.png`, `03-baseline_test_metrics_summary.png`
- `04-transformer_training_history.png` — loss/accuracy/macro-F1 görbék
- `04-transformer_test_report.json` — test metrikák (Accuracy, Macro/Weighted F1, MAE, RMSE, per-class)

### `output/reports/` (evaluation)
- `05-evaluation_test_report.json` — részletes metrikák (Accuracy, Macro/Weighted F1, MAE, RMSE)
- `05-evaluation_test_confusion_matrix.png` — confusion matrix

### `output/reports/` (advanced)
- `advanced/robustness/robustness_results.json` — robusztussági tesztek (Accuracy, Macro/Weighted F1)
- `advanced/robustness/robustness_accuracy.png` — összehasonlító ábra
- `advanced/explainability/attention_importance.json` — attention-alapú token fontosság
- `advanced/explainability/attention_summary.json` — osztályonkénti összegzések
- `advanced/explainability/misclassification_analysis.json` — hibaanalízis
- `advanced/explainability/confusion_pairs.png` — leggyakoribb félreosztások

---

## 🌐 ML Service - API + GUI

### Opció 1: Pipeline részeként (automatikus indítás) ⭐

**Csak API:**
```bash
# Docker Compose (ajánlott)
docker-compose up training-with-api

# vagy környezeti változóval
export START_API_SERVICE=1  # Linux/macOS
$env:START_API_SERVICE=1    # Windows PowerShell
```

**API + Frontend (teljes stack):**
```bash
# Automatikus script (legegyszerűbb)
.\docker-run-full-stack.ps1  # Windows
bash docker-run-full-stack.sh  # Linux/macOS

# vagy Docker Compose
docker-compose up training-full-stack

# vagy környezeti változókkal
export START_API_SERVICE=1  # Linux/macOS
export START_FRONTEND_SERVICE=1
$env:START_API_SERVICE=1    # Windows PowerShell
$env:START_FRONTEND_SERVICE=1
```

Az API és Frontend automatikusan elindul a training befejezése után:
- API: port 8000
- Frontend: port 8501

### Opció 2: Külön indítás (már kész modellekkel)

**REST API (FastAPI):**

```bash
# Docker Compose
docker-compose up api

# vagy manuálisan
cd src
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Web GUI (Streamlit):**

```bash
# Docker Compose
docker-compose up frontend

# vagy manuálisan
cd src
streamlit run frontend/app.py
```

Böngészőben: 
- API: [http://localhost:8000](http://localhost:8000)
- GUI: [http://localhost:8501](http://localhost:8501)

### API Használat

**Endpoint:** `POST /predict`

**Példa request:**
```json
{
  "text": "A szerződés hatálya visszamenőleg nem érvényesíthető...",
  "model_type": "transformer"
}
```

**Példa response:**
```json
{
  "prediction": "3-Többé/kevésbé megértem",
  "confidence": 0.89,
  "model_used": "transformer"
}
```


---

## 🛠️ Hibaelhárítás

### GPU támogatás hiányzik

**Probléma:** `RuntimeError: No CUDA GPUs are available`

**Megoldás:**
1. Ellenőrizd az NVIDIA driver-t: `nvidia-smi`
2. Docker Desktop → Settings → Resources → WSL Integration → Enable
3. Futtatás `--gpus all` flag-gel

### Memóriahiány (OOM)

**Tünetek:** Docker crash vagy `OutOfMemoryError`

**Megoldás:**
- Csökkentsd a `TRANSFORMER_BATCH_SIZE` értékét (pl. `4` helyett `2`)
- Docker Desktop → Settings → Resources → növeld a memória limitet (min. 8GB ajánlott)


### Encoding hiba a CSV-kben

**Probléma:** `UnicodeDecodeError`

**Megoldás:** Minden fájlírás a pipeline-ban már `encoding='utf-8'` paraméterrel történik. Ha lokálisan olvasod be, használj `encoding='utf-8-sig'` paramétert.


### Lassú futás CPU-n

A Transformer fine-tuning CPU-n 6+ óra is lehet. A baseline modell (~5 perc) működik CPU-n is, de a fő pipeline a transformer modellt használja.

**Opció:** Használd csak a baseline modellt (03) vagy bérelj GPU-s cloud instance-t (Google Colab, AWS, Azure).

---


## 📌 Megjegyzések

- A pipeline szekvenciálisan fut a `run.sh` szerint (01→07), baseline opcionális.
- Early stopping a `val_macro_f1`-t figyeli; a legjobb checkpoint automatikusan mentésre kerül.
- Az advanced statisztikák (olvashatóság, diverzitás, TF-IDF, korreláció) magyar jogi szövegekre optimalizáltak.
- A deduplikáció és címke-szűrés csak EDA-célú; a `raw_dataset.csv` változatlan marad.
- 05–07 minden értékelést a transformer modellel végez (batch inference, robustness, explainability).
- **Minden fájlírás és olvasás explicit UTF-8 encodinggal történik a magyar karakterek miatt.**

---


## 📄 Licenc

MIT License — lásd `LICENSE` fájl.

## 👤 Szerző

NagypalMarton — [GitHub](https://github.com/NagypalMarton/DeepLearning_Project-Legal_Text_Decoder)