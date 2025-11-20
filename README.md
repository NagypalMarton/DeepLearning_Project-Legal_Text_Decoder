# Legal Text Decoder

NLP rendszer jogi szövegek (ÁSZF/ÁFF) érthetőségének automatikus értékelésére (1-5 skála).
Docker + PyTorch + GPU támogatás | Cross-platform

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

### Automatikus platform detektálás

```bash
# Windows PowerShell
.\docker-run.ps1

# Linux/macOS/Git Bash
bash docker-run.sh
```

### Manuális Docker futtatás

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

Megjegyzés: A futás a teljes 01→07 pipeline-t végigviszi. A baseline (03) opcionális.

---

## 🎯 Követelmény-Fájl Megfeleltetés

| # | Outstanding Level Követelmény | Implementáció | Fájl |
|---|-------------------------------|---------------|------|
| 1 | **Containerization** | Docker + GPU támogatás | `Dockerfile` |
| 2 | **Data acquisition and analysis** | JSON parser, RAW EDA, advanced statistics | `01_data_acquisition_and_analysis.py` |
| 3 | **Data cleansing and preparation** | Text cleaning, deduplication, stratified split | `02_data_cleansing_and_preparation.py` |
| 4 | **Defining evaluation criteria** | Transformer (HuBERT) batch inference, metrics, confusion matrix | `05_defining_evaluation_criteria.py` |
| 5 | **Baseline model (opcionális)** | TF-IDF + LogisticRegression | `03_baseline_model.py` |
| 6 | **Incremental model development** | Transformer (HuBERT) fine-tuning | `04_incremental_model_development.py` |
| 7 | **Advanced evaluation** | Transformer-based Robustness + Explainability | `06_advanced_evaluation_robustness.py` <br> `07_advanced_evaluation_explainability.py` |
| 8 | **ML as a service** | REST API + Web GUI | `src/api/app.py` <br> `src/frontend/app.py` |

---

## 📋 Pipeline Lépések

Névkonvenció a kimenetekre: minden mérési/ábra/riport fájl név elején lépés-prefix szerepel.
Minta: `{lépés}-{rövid_név}_{típus}_{split}.{ext}`
Példák: `01-acquisition_raw_eda_statistics.txt`, `03-baseline_test_confusion_matrix.png`, `06-robustness_results.json`.

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
**Cél:** Transformer (HuBERT) fine-tuning, feature fusion, ordinal label mapping, legjobb checkpoint mentése

**Fő fejlesztések:**
- Readability feature fusion (MLP branch, standardized)
- Mean pooling (stabilabb, mint CLS)
- Sqrt-scaled osztálysúlyok, label smoothing
- Ordinal label mapping (1–5 skála)
- **CORAL ordinal regression loss támogatás** (opcionális, `USE_CORAL=1`)
- Early stopping, checkpoint mentés

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

**CORAL loss (opcionális):**
- További MAE/RMSE csökkenés, Macro F1 javulás várható
- Aktiválás: `USE_CORAL=1` környezeti változóval

### 5. 05_defining_evaluation_criteria.py
**Cél:** Transformer batch inference, metrikák, confusion matrix

**Kimenetek:**
- `output/reports/05-evaluation_test_report.json`
- `output/reports/05-evaluation_test_confusion_matrix.png`

### 6. 06_advanced_evaluation_robustness.py
**Cél:** Transformer robustness tesztek (zaj, csonkítás, stb.)

**Kimenetek:**
- `output/reports/06-robustness_results.json`
- `output/reports/06-robustness_comparison.png`

### 7. 07_advanced_evaluation_explainability.py
**Cél:** Transformer attention-alapú magyarázhatóság, hibaanalízis, confusion pairs

**Kimenetek:**
- `output/reports/07-explainability_attention_importance.json`
- `output/reports/07-explainability_attention_summary.json`
- `output/reports/07-explainability_misclassification_analysis.json`
- `output/reports/07-explainability_top_confusion_pairs.png`

> A `src/run.sh` sorban futtatja az összes `src/*.py` fájlt (01→07). Dockerben ez az alapértelmezett belépési pont. A baseline (03) opcionális; a fő pipeline a transformer modellt használja minden értékeléshez.

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

### `output/reports/` (robustness)
- `06-robustness_results.json` — robusztussági tesztek eredményei (Macro/Weighted F1)
- `06-robustness_comparison.png` — összehasonlító ábra

### `output/reports/` (explainability)
- `07-explainability_attention_importance.json` — attention-alapú token fontosság
- `07-explainability_attention_summary.json` — osztályonkénti összegzések
- `07-explainability_misclassification_analysis.json` — hibaanalízis
- `07-explainability_top_confusion_pairs.png` — leggyakoribb félreosztások

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


### Web GUI (Streamlit)

```bash
# Frontend indítása (port: 8501)
docker-compose up frontend
# vagy
cd src
streamlit run frontend/app.py
```

Böngészőben: [http://localhost:8501](http://localhost:8501)

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

A Transformer fine-tuning CPU-n 6+ óra is lehet. A baseline modell (~5 perc) működik CPU-n is, de a fő pipeline a transformer modellt használja.

**Opció:** Használd csak a baseline modellt (03) vagy bérelj GPU-s cloud instance-t (Google Colab, AWS, Azure).

---

## 📌 Megjegyzések

- A pipeline szekvenciálisan fut a `run.sh` szerint (01→07), baseline opcionális.
- Early stopping a `val_macro_f1`-t figyeli; a legjobb checkpoint automatikusan mentésre kerül.
- Az advanced statisztikák (olvashatóság, diverzitás, TF-IDF, korreláció) magyar jogi szövegekre optimalizáltak.
- A deduplikáció és címke-szűrés csak EDA-célú; a `raw_dataset.csv` változatlan marad.
- 05–07 minden értékelést a transformer modellel végez (batch inference, robustness, explainability).

---

## 📄 Licenc

MIT License — lásd `LICENSE` fájl.

## 👤 Szerző

NagypalMarton — [GitHub](https://github.com/NagypalMarton/DeepLearning_Project-Legal_Text_Decoder)