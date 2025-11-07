# Legal Text Decoder

NLP rendszer jogi szövegek (ÁSZF/ÁFF) érthetőségének automatikus értékelésére (1-5 skála). Docker + PyTorch + GPU támogatás.

## 📚 Tartalomjegyzék

- [Gyors Indítás](#-gyors-indítás)
- [Fő lépések (pipeline)](#-fő-lépések-pipeline)
- [Adatformátum](#-adatformátum)
- [Környezeti változók](#-környezeti-változók)
- [Kimenetek](#-kimenetek)
- [Hibaelhárítás](#-hibaelhárítás)

## Fő lépések (pipeline)

1. **01_data_processing.py** — JSON adatok betöltése (fájl vagy mappa), szöveg tisztítás, label kinyerés, stratifikált train/val/test split és mentés CSV-be az OUTPUT_DIR/processed mappába.
2. **02_feature_engineering.py** — Egyszerű szövegstatisztikák (word_count, avg_word_len) hozzáadása és opcionális Sentence-BERT beágyazások mentése az OUTPUT_DIR/features mappába.
3. **03_train_baseline.py** — Baseline szövegklasszifikációs modell: TF‑IDF + LogisticRegression. Modell mentése (OUTPUT_DIR/models), metrikák mentése (OUTPUT_DIR/reports).
4. **04_train_transformer.py** — Transformer alapú modell (pl. HuBERT) finomhangolása a jogi szövegeken. GPU ajánlott! Modell és tokenizer mentése (OUTPUT_DIR/models/transformer_model).
5. **05_evaluation.py** — Külön értékelő script a baseline modellre a test spliten (OUTPUT_DIR/evaluation).
6. **06_robustness_tests.py** — Robusztussági tesztek: zajjal és csonkolással módosított szövegeken értékeli a baseline modellt (OUTPUT_DIR/robustness).
7. **07_explainability.py** — Modell értelmezhetőség: top feature-ök osztályonként, predikció magyarázatok, hibaelemzés (OUTPUT_DIR/explainability).

> A `src/run.sh` sorban futtatja az összes `src/*.py` fájlt (ábécérendben). Dockerben ez az alapértelmezett belépési pont.

## Adatformátum (JSON)

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

## Környezeti változók

**Adatkezelés:**
- `DATA_DIR` — Bemeneti adat mappa (alap: `/app/data` Dockerben).
- `OUTPUT_DIR` — Kimeneti mappa (alap: `/app/output`).

**Baseline modell (TF-IDF + LogisticRegression):**
- `TFIDF_MAX_FEATURES` — TF‑IDF max jellemzők száma (alap: 20000).
- `TFIDF_NGRAM_MAX` — TF‑IDF n-gram felső határ (alap: 2).
- `LR_C` — LogisticRegression C paramétere (alap: 1.0).

**Transformer modell:**
- `TRANSFORMER_MODEL` — Használandó transformer modell neve (alap: `SZTAKI-HLT/hubert-base-cc`).
- `BATCH_SIZE` — Batch méret a tanításhoz (alap: 8).
- `EPOCHS` — Tanítási epochok száma (alap: 3).
- `LEARNING_RATE` — Tanulási ráta (alap: 2e-5).
- `MAX_LENGTH` — Maximális szekvencia hossz tokenizáláskor (alap: 512).

**Feature engineering:**
- `ENABLE_EMBEDDINGS` — Ha `true`, Sentence‑BERT beágyazások számítása a 02-es lépésben (alap: false).
- `EMBEDDING_MODEL` — Embedding modell neve (alap: `paraphrase-multilingual-MiniLM-L12-v2`).

## Futtatás Dockerrel

1) Image build:

```powershell
docker build -t deeplearning_project-legal_text_decoder:1.0 .
```

2) Konténer futtatása (PowerShell, GPU-val és volumekkel):

```powershell
docker run --rm --gpus all `
	-v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
	-v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
	deeplearning_project-legal_text_decoder:1.0 > training_log.txt 2>&1
```

Az összes kimenet az `C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output` mappában lesz elérhető (Windows host oldalon).

## Lokális futtatás (opcionális)

Python környezetben (a `requirements.txt` telepítése után) egyenként is futtathatók a scriptek:

```powershell
$env:DATA_DIR = "C:\\path\\to\\data"; $env:OUTPUT_DIR = "C:\\path\\to\\output"; python src/01_data_processing.py
python src/02_feature_engineering.py
python src/03_train_baseline.py
python src/05_evaluation.py
```

## Kimenetek

- `OUTPUT_DIR/processed/` — `train.csv`, `val.csv`, `test.csv` (vagy `processed_data.csv` fallback esetén) szövegstatisztikákkal kiegészítve
- `OUTPUT_DIR/features/` — szövegstatisztika ábrák (hisztogramok), opcionális `embeddings_*.npy` és `embeddings_meta.json`
- `OUTPUT_DIR/models/` — `baseline_model.pkl` (TF-IDF + LogReg), `transformer_model/` (finomhangolt transformer), `label_mapping.json`
- `OUTPUT_DIR/reports/` — baseline és transformer metrikák (val/test JSON riportok), `transformer_training_history.png`
- `OUTPUT_DIR/evaluation/` — külön teszt riport és konfúziós mátrix a baseline modellhez
- `OUTPUT_DIR/robustness/` — robusztussági tesztek eredményei (`robustness_results.json`, `robustness_comparison.png`)
- `OUTPUT_DIR/explainability/` — feature importance, predikció magyarázatok, hibaelemzés JSON-ben és ábrákban

## Megjegyzések és ismert korlátok

- A stratifikált split legalább két osztályt és elegendő mintát igényel osztályonként. Kevés minta esetén hibaüzenetet kaphatsz.
- A Sentence‑BERT beágyazások letöltése internetet és több memóriát igényelhet; alapértelmezetten ki van kapcsolva.
- A **transformer modell tanítása (04_train_transformer.py) GPU-t igényel** a hatékony futáshoz. CPU-n is fut, de sokkal lassabb.
- A transformer modell alapértelmezetten a magyar **HuBERT** modellt használja, de ez környezeti változóval módosítható más modellekre (pl. `bert-base-multilingual-cased`).
- Ha csak a baseline modellt szeretnéd futtatni (gyorsabb, kevesebb erőforrás), egyszerűen töröld vagy nevezd át a `04_train_transformer.py` fájlt a pipeline előtt.

## 🚀 Gyors Indítás

```powershell
# 1. Build
docker build -t deeplearning_project-legal_text_decoder:1.0 .

# 2. Futtatás (GPU-val)
docker run --rm --gpus all `
  -v "C:\path\to\data:/app/data" `
  -v "C:\path\to\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > training_log.txt 2>&1
```

**Futási idő:** ~45-60 perc GPU-val | ~6+ óra CPU-n (transformer miatt)

## 📋 Fő lépések (pipeline)

1. **01_data_processing.py** — JSON adatok betöltése (fájl vagy mappa), szöveg tisztítás, label kinyerés, stratifikált train/val/test split (60/20/20) és mentés CSV-be az OUTPUT_DIR/processed mappába.
2. **02_feature_engineering.py** — Egyszerű szövegstatisztikák (word_count, avg_word_len) hozzáadása és opcionális Sentence-BERT beágyazások mentése az OUTPUT_DIR/features mappába.
3. **03_train_baseline.py** — Baseline szövegklasszifikációs modell: TF‑IDF + LogisticRegression. Modell mentése (OUTPUT_DIR/models), metrikák mentése (OUTPUT_DIR/reports).
4. **04_train_transformer.py** — Transformer alapú modell (pl. HuBERT) finomhangolása a jogi szövegeken. GPU ajánlott! Modell és tokenizer mentése (OUTPUT_DIR/models/transformer_model).
5. **05_evaluation.py** — Külön értékelő script a baseline modellre a test spliten (OUTPUT_DIR/evaluation).
6. **06_robustness_tests.py** — Robusztussági tesztek: zajjal és csonkolással módosított szövegeken értékeli a baseline modellt (OUTPUT_DIR/robustness).
7. **07_explainability.py** — Modell értelmezhetőség: top feature-ök osztályonként, predikció magyarázatok, hibaelemzés (OUTPUT_DIR/explainability).

> A `src/run.sh` sorban futtatja az összes `src/*.py` fájlt (ábécérendben). Dockerben ez az alapértelmezett belépési pont.

## 📄 Adatformátum

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

**Megjegyzés:** Az első elem első választása kerül felhasználásra: `annotations[0].result[0].value.choices[0]`

```
output/
├── processed/
│   ├── train.csv               # Training set (60%, szövegstatisztikákkal)
│   ├── val.csv                 # Validation set (20%)
│   └── test.csv                # Test set (20%)
│
├── features/
│   ├── train_word_count_hist.png
│   ├── train_avg_word_len_hist.png
│   ├── embeddings_train.npy    (ha ENABLE_EMBEDDINGS=true)
│   ├── embeddings_val.npy
│   ├── embeddings_test.npy
│   └── embeddings_meta.json
│
├── models/
│   ├── baseline_model.pkl      # Sklearn pipeline
│   ├── label_mapping.json      # Label → ID mapping
│   └── transformer_model/      # HuBERT modell
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

## ⚙️ Környezeti változók

### Alapvető
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `DATA_DIR` | `/app/data` | Input adatok helye |
| `OUTPUT_DIR` | `/app/output` | Kimenetek helye |

### Baseline
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `TFIDF_MAX_FEATURES` | `20000` | Max TF-IDF feature-ök száma |
| `TFIDF_NGRAM_MAX` | `2` | N-gram felső határ |
| `LR_C` | `1.0` | Regularizációs paraméter |

### Transformer
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `TRANSFORMER_MODEL` | `SZTAKI-HLT/hubert-base-cc` | Transformer modell név |
| `BATCH_SIZE` | `8` | Batch méret (8GB VRAM-hoz) |
| `EPOCHS` | `3` | Epochok száma |
| `LEARNING_RATE` | `2e-5` | Tanulási ráta |
| `MAX_LENGTH` | `512` | Max token hossz |

### Embeddings
| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `ENABLE_EMBEDDINGS` | `false` | Sentence-BERT embeddings be/ki |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding modell |

## � Kimenetek

```
output/
├── processed/          # train/val/test CSV-k
├── features/           # Statisztikák, embeddings
├── models/             # baseline_model.pkl, transformer_model/
├── reports/            # Metrikák, confusion matrix
├── evaluation/         # Test eredmények
├── robustness/         # Robusztussági tesztek
└── explainability/     # Feature importance
```

## 🐛 Hibaelhárítás

| Probléma | Megoldás |
|----------|----------|
| `CUDA not available` | Ellenőrizd: `nvidia-smi`, Docker GPU support |
| `CUDA out of memory` | `-e BATCH_SIZE=4` vagy `-e MAX_LENGTH=256` |
| Stratified split hiba | Min. 3-5 példa/osztály szükséges |
| Lassú futás CPU-n | Használj GPU-t vagy töröld `04_train_transformer.py` |
| JSON parsing hiba | Ellenőrizd az Adatformátum szekciót |

## ⏱️ Teljesítmény

| Modell | Accuracy | Training | GPU | Memory |
|--------|----------|----------|-----|--------|
| Baseline | 60-75% | ~3 min | Nem kell | ~500 MB |
| Transformer | 70-85% | ~40 min | 8GB+ VRAM | ~2-4 GB |

**Teljes pipeline:** ~45-60 min (GPU) | ~6+ óra (CPU)

## 📝 Licenc

Lásd: `LICENSE`
