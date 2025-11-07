# DeepLearning_Project-Legal_Text_Decoder

A projekt célja egy olyan természetes nyelvfeldolgozási (Natural Language Processing, NLP) modell létrehozása, amely képes megjósolni, hogy egy adott Általános Szerződési Feltételek (ÁSZF) és/vagy Általános Felhasználási Feltételek (ÁFF) szövegének egy bekezdése mennyire könnyen vagy nehezen érthető egy átlagos felhasználó számára.

## Fő lépések (pipeline)

1. 01_data_processing.py — JSON adatok betöltése (fájl vagy mappa), szöveg tisztítás, label kinyerés, stratifikált train/val/test split és mentés CSV-be az OUTPUT_DIR/processed mappába.
2. 02_feature_engineering.py — Egyszerű szövegstatisztikák (word_count, avg_word_len) hozzáadása és opcionális Sentence-BERT beágyazások mentése az OUTPUT_DIR/features mappába.
3. 03_train_baseline.py — Baseline szövegklasszifikációs modell: TF‑IDF + LogisticRegression. Modell mentése (OUTPUT_DIR/models), metrikák mentése (OUTPUT_DIR/reports).
4. 05_evaluation.py — Külön értékelő script a mentett modellre a test spliten (OUTPUT_DIR/evaluation).

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

- `DATA_DIR` — Bemeneti adat mappa (alap: `/app/data` Dockerben).
- `OUTPUT_DIR` — Kimeneti mappa (alap: `/app/output`).
- `TFIDF_MAX_FEATURES` — TF‑IDF max jellemzők száma (alap: 20000).
- `TFIDF_NGRAM_MAX` — TF‑IDF n-gram felső határ (alap: 2).
- `LR_C` — LogisticRegression C paramétere (alap: 1.0).
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

- `OUTPUT_DIR/processed/` — `train.csv`, `val.csv`, `test.csv` (vagy `processed_data.csv` fallback esetén)
- `OUTPUT_DIR/features/` — szövegstatisztika ábrák, opcionális `embeddings_*.npy`
- `OUTPUT_DIR/models/` — `baseline_model.pkl`
- `OUTPUT_DIR/reports/` — baseline metrikák (val/test)
- `OUTPUT_DIR/evaluation/` — külön teszt riport és konfúziós mátrix

## Megjegyzések és ismert korlátok

- A stratifikált split legalább két osztályt és elegendő mintát igényel osztályonként. Kevés minta esetén hibaüzenetet kaphatsz.
- A Sentence‑BERT beágyazások letöltése internetet és több memóriát igényelhet; alapértelmezetten ki van kapcsolva.

## Licenc

Lásd: `LICENSE`.
