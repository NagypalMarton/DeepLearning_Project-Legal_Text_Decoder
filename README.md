# Legal Text Decoder

NLP rendszer jogi szövegek (ÁSZF/ÁFF) érthetőségének automatikus értékelésére (1-5 skála).  
**Docker + PyTorch + GPU támogatás | Cross-platform**

## 📚 Tartalomjegyzék

- [Gyors Indítás](#-gyors-indítás)
- [Követelmény-Fájl Megfeleltetés](#-követelmény-fájl-megfeleltetés)
- [Pipeline Lépések](#-pipeline-lépések)
- [Statisztikai Elemzések](#-statisztikai-elemzések)
- [Adatformátum](#-adatformátum)
- [Környezeti Változók](#-környezeti-változók)
- [Kimenetek](#-kimenetek)
- [ML Service - API + GUI](#-ml-service---api--gui)
- [Hibaelhárítás](#-hibaelhárítás)

---

## 🚀 Gyors Indítás

### Automatikus platform detektálás:

```bash
# Windows PowerShell
.\docker-run.ps1

# Linux/macOS/Git Bash
bash docker-run.sh
```

### Manuális Docker futtatás:

```bash
# 1. Build
docker build -t deeplearning_project-legal_text_decoder:1.0 .

# 2. Futtatás (Windows PowerShell)
docker run --rm --gpus all `
  -v "${PWD}\attach_folders\data:/app/data" `
  -v "${PWD}\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0

# 2. Futtatás (Linux/macOS)
docker run --rm --gpus all \
  -v "$(pwd)/attach_folders/data:/app/data" \
  -v "$(pwd)/attach_folders/output:/app/output" \
  deeplearning_project-legal_text_decoder:1.0
```

**Futási idő:** ~5-10 perc GPU-val | ~20-30 perc CPU-n

---

## 🎯 Követelmény-Fájl Megfeleltetés

| # | Outstanding Level Követelmény | Implementáció | Fájl |
|---|-------------------------------|---------------|------|
| 1 | **Containerization** | Docker + GPU támogatás | `Dockerfile` |
| 2 | **Data acquisition and analysis** | JSON parser, RAW EDA, advanced statistics | `01_data_acquisition_and_analysis.py` |
| 3 | **Data cleansing and preparation** | Text cleaning, deduplication, stratified split | `02_data_cleansing_and_preparation.py` |
| 4 | **Defining evaluation criteria** | Metrics, confusion matrix | `05_defining_evaluation_criteria.py` |
| 5 | **Baseline model** | TF-IDF + LogisticRegression | `03_baseline_model.py` |
| 6 | **Incremental model development** | Transformer (HuBERT) fine-tuning | `04_incremental_model_development.py` |
| 7 | **Advanced evaluation** | Robustness + Explainability | `06_advanced_evaluation_robustness.py` <br> `07_advanced_evaluation_explainability.py` |
| 8 | **ML as a service** | REST API + Web GUI | `src/api/app.py` <br> `src/frontend/app.py` |

---

## 📋 Pipeline Lépések

### 1. **01_data_acquisition_and_analysis.py**
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
- `output/raw/removed_duplicates.csv` — eltávolított duplikátumok listája
- `output/raw/removed_missing_labels.csv` — eltávolított hiányzó címkés sorok
- `output/features/raw_eda_statistics.txt` — statisztikai összefoglaló
- `output/features/raw_label_distribution.png` — label eloszlás
- `output/features/*_by_label.png` — metrikák boxplot-jai címkénként (6 db)
- `output/features/tfidf_top_words_by_label.csv` — jellemző szavak táblázat
- `output/features/correlation_matrix.png` — korrelációs heatmap

### 2. **02_data_cleansing_and_preparation.py**
**Cél:** Szövegtisztítás és train/val/test split

**Funkciók:**
- Betölti a szűrt EDA adatokat (`raw_dataset_eda_filtered.csv`)
- Unicode normalizálás, whitespace kezelés
- Speciális karakterek szűrése (magyar jogi szövegekre optimalizálva)
- Stratified split (60% train, 20% val, 20% test)
- Szövegstatisztikák hozzáadása (word_count, avg_word_len)
- Opcionális: Sentence-BERT embeddings

**Kimenetek:**
- `output/processed/train.csv`
- `output/processed/val.csv`
- `output/processed/test.csv`
- `output/features/clean_word_count_hist.png`
- `output/features/clean_avg_word_len_hist.png`

### 3. **03_baseline_model.py**
**Cél:** Baseline szövegklasszifikáció

**Modell:** TF-IDF (max_features=20000, ngram_range=(1,2)) + LogisticRegression (C=1.0)

**Kimenetek:**
- `output/models/baseline_model.pkl`
- `output/reports/baseline_val_report.json`
- `output/reports/baseline_test_report.json`
- `output/reports/baseline_test_confusion_matrix.png`

### 4-7. **További pipeline lépések**
- **04**: Transformer modell (HuBERT) fine-tuning
- **05**: Evaluation criteria
- **06**: Robustness testing
- **07**: Explainability (SHAP, feature importance)

> A `src/run.sh` sorban futtatja az összes `src/*.py` fájlt. Dockerben ez az alapértelmezett belépési pont.

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
- `DATA_DIR` — Bemeneti adat mappa (alap: `/app/data` Dockerben)
- `OUTPUT_DIR` — Kimeneti mappa (alap: `/app/output`)

**Baseline modell (TF-IDF + LogisticRegression):**
- `TFIDF_MAX_FEATURES` — TF-IDF max jellemzők száma (alap: `20000`)
- `TFIDF_NGRAM_RANGE` — N-gram tartomány (alap: `1,2`)
- `LOGREG_C` — Regularizációs paraméter (alap: `1.0`)

**Embeddings (opcionális):**
- `ENABLE_EMBEDDINGS` — Sentence-BERT embeddings számítása (alap: `false`)
- `EMBEDDING_MODEL` — Használt modell neve (alap: `paraphrase-multilingual-MiniLM-L12-v2`)

**Transformer modell:**
- `TRANSFORMER_MODEL` — Használt modell (alap: `SZTAKI-HLT/hubert-base-cc`)
- `TRANSFORMER_EPOCHS` — Training epoch-ok száma (alap: `3`)
- `TRANSFORMER_BATCH_SIZE` — Batch méret (alap: `8`)
- `TRANSFORMER_LR` — Learning rate (alap: `2e-5`)

---

## 📦 Kimenetek

### `output/raw/`
- `raw_dataset.csv` — teljes nyers adathalmaz (minden sor, változatlan)
- `raw_dataset_eda_filtered.csv` — deduplikált és címke-szűrt snapshot (pipeline input)
- `raw_dataset_eda_enhanced.csv` — összes statisztikai metrikával bővített adathalmaz
- `removed_duplicates.csv` — eltávolított duplikátumok listája (227 sor)
- `removed_missing_labels.csv` — üres choices vagy text sorok listája (136 sor)

### `output/features/`
**RAW EDA:**
- `raw_eda_statistics.txt` — duplikációs és szűrési statisztikák szöveges összefoglalója
- `raw_label_distribution.png` — besorolások eloszlása (bar chart)
- `raw_word_count_hist.png` — nyers szöveghosszok eloszlása
- `raw_avg_word_len_hist.png` — nyers átlagos szóhosszok eloszlása

**Advanced Statistics (címkénkénti boxplotok):**
- `flesch_score_by_label.png` — Flesch Reading Ease
- `fog_index_by_label.png` — Gunning Fog Index
- `smog_index_by_label.png` — SMOG Index
- `ttr_by_label.png` — Type-Token Ratio
- `mattr_by_label.png` — Moving Average TTR
- `hapax_ratio_by_label.png` — Hapax legomena arány

**Analitikai kimenet:**
- `tfidf_top_words_by_label.csv` — legjellemzőbb szavak minden címkére
- `correlation_matrix.png` — feature korrelációs heatmap

**CLEAN EDA:**
- `clean_word_count_hist.png` — tisztított szöveg szógyakoriság
- `clean_avg_word_len_hist.png` — tisztított szöveg szóhosszúság

### `output/processed/`
- `train.csv` (2022 sor, ~60%)
- `val.csv` (675 sor, ~20%)
- `test.csv` (675 sor, ~20%)

Minden CSV oszlopai: `text`, `label`, `word_count`, `avg_word_len`

### `output/models/`
- `baseline_model.pkl` — TF-IDF + LogisticRegression
- `transformer_model/` — HuBERT checkpoint + tokenizer

### `output/reports/`
- `baseline_val_report.json` — validation metrikák
- `baseline_test_report.json` — test metrikák (accuracy, balanced_accuracy, macro F1, per-class metrics)
- `baseline_test_confusion_matrix.png` — confusion matrix heatmap

### `output/evaluation/`, `output/robustness/`, `output/explainability/`
További értékelési kimenetek a 05-07 lépésekből.

---

## 🌐 ML Service - API + GUI

### REST API (FastAPI)

```bash
# API indítása (port: 8000)
cd src
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Endpoint:** `POST /predict`

**Példa request:**
```json
{
  "text": "A szerződés hatálya visszamenőleg nem érvényesíthető..."
}
```

**Példa response:**
```json
{
  "prediction": "3-Többé/kevésbé megértem",
  "confidence": 0.89,
  "model": "baseline"
}
```

### Web GUI (Streamlit)

```bash
# Frontend indítása (port: 8501)
cd src
streamlit run frontend/app.py
```

Böngészőben: `http://localhost:8501`

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

**Megoldás:** A scriptek már UTF-8-sig encoding-ot használnak. Ha lokálisan olvasod be, használj `encoding='utf-8-sig'` paramétert.

### Lassú futás CPU-n

A Transformer fine-tuning CPU-n 6+ óra is lehet. Baseline modell (~5 perc) működik CPU-n is.

**Opció:** Használd csak a baseline modellt vagy bérelj GPU-s cloud instance-t (Google Colab, AWS, Azure).

---

## 📌 Megjegyzések

- A pipeline **szekvenciálisan fut** a `run.sh` által meghatározott sorrendben
- Minden script **függetlenül futtatható** manuálisan is lokális környezetben (Docker nélkül)
- Az **advanced statistics** (olvashatóság, diverzitás, TF-IDF, korreláció) kifejezetten **magyar jogi szövegekre** vannak optimalizálva
- A **deduplikáció és címke-szűrés** csak EDA-célú; a `raw_dataset.csv` változatlan marad

---

## 📄 Licenc

MIT License — lásd `LICENSE` fájl.

## 👤 Szerző

NagypalMarton — [GitHub](https://github.com/NagypalMarton/DeepLearning_Project-Legal_Text_Decoder)