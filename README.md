# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Nagypál Márton Péter
- **Aiming for +1 Mark**: Yes

### Solution Description

Ez a projekt jogi szövegek érthetőségének automatikus értékelését oldja meg gépi tanulás segítségével, 1-5 skálán (1: nagyon nehezen érthető, 5: könnyen érthető).

**Probléma**: Jogi szövegek gyakran nehezen érthetőek a civil olvasók számára. A cél egy olyan rendszer kifejlesztése, amely automatikusan értékeli egy bekezdés olvashatóságát.

**Modell architektúra**: A megoldás HuBERT (Hungarian BERT) transzformerre épül, progresszív modellfejlesztéssel:
1. **Step1_Baseline**: Transzformer + egyszerű lineáris osztályozó (CLS pooling)
2. **Step2_Extended**: 3-rétegű adapter BatchNorm-al és rétegenkénti dropout-tal (0.3→0.25→0.2)
3. **Step3_Advanced**: Attention pooling + mély adapter + gating mechanizmus
4. **Final_Balanced**: Production-ready, mean pooling + kiegyensúlyozott architektúra (AJÁNLOTT)

**Tréning módszertan**:
- Stratifikált train/val/test split (60/20/20)
- Class-weighted CrossEntropyLoss (sqrt-scaled súlyozás)
- Label smoothing (0.02), korai megállás (patience=3)
- Overfitting sanity check a baseline modellen (single batch, 100% accuracy cél)
- Optimalizáció: AdamW, linear warmup scheduler, gradient accumulation, mixed precision
- Explorációs stratégia: 33%-os subset-en 4 modell gyors kipróbálása, majd a győztes teljes adaton való újratanítása

**Értékelés**: 
- Osztályozási metrikák: accuracy, precision, recall, F1 (macro/weighted)
- Ordinális regressziós metrikák: MAE, RMSE (az 1-5 skála miatt)
- Speciális: ROC AUC, log loss, weighted FN cost (súlyosabb osztályok FN-jére magasabb büntetés)
- Robusztusság tesztelés: zaj (5%, 10%, 20%), csonkolás (75%, 50%, 25%)
- Magyarázhatóság: attention-based token importance analízis, misclassification párok

**Eredmények**: A modellek validation accuracy-je ~75-85%, MAE < 0.5. A Final_Balanced modell biztosítja a legjobb generalizációt kis train-val gap-pel.

### Extra Credit Justification

A projekt az Outstanding Level minden követelményét teljesíti és az alábbi innovatív megoldásokat tartalmazza:

1. **Progresszív modellfejlesztés**: 4 különböző architektúra szisztematikus összehasonlítása (Baseline → Extended → Advanced → Final Balanced) automatizált pipeline-ban, részletes összehasonlító vizualizációkkal.

2. **Haladó értékelési módszerek**:
   - Robusztusság tesztelés (zaj, csonkolás) automatizált perturbációs tesztekkel
   - Attention-based explainability (top-k token fontosság osztályonként)
   - Konfúziós pár elemzés hibák diagnosztizálásához

3. **Production-ready ML szolgáltatás**:
   - FastAPI backend automatikus modell betöltéssel és validációval
   - Streamlit frontend interaktív vizualizációkkal (osztály valószínűségek, confidence)
   - Konténerizált deployment (API + Frontend egy stack-ben)

4. **Automatizált adatgyűjtés**: SharePoint integráció automatikus ZIP letöltéssel és kicsomagolással (fallback mechanizmussal).

5. **Részletes metrikák és vizualizációk**:
   - Readability metrikák (Flesch, Gunning Fog, SMOG, TTR, MATTR, hapax ratio)
   - TF-IDF top words osztályonként
   - Korrelációs mátrix feature-ök között
   - Training history plots minden modellre
   - Per-class precision/recall/F1/support bar chart-ok

6. **Overfitting sanity check**: A baseline modell automatikusan ellenőrzi, hogy képes-e 100% accuracy-re egyetlen batch-en, mielőtt teljes tréningbe kezd (early bug detection).

7. **Explorációs stratégia**: Subset-alapú gyors kísérletezés (33% adat), majd a győztes modell teljes adaton való retrain-je, jelentős időmegtakarítással.

### Data Preparation

**Adatforrás**: Label Studio JSON formátumú annotált jogi szövegek SharePoint-ról.

**Automatikus feldolgozás**: A `src/01_data_acquisition_and_analysis.py` script automatikusan letölti és kicsomagolja az adatokat a SharePoint linkről. Ha a `DATA_DIR` könyvtár üres, a script:
1. Letölti a ZIP fájlt SharePoint-ról (több URL-lel fallback)
2. Kicsomagolja és kimásolja a JSON fájlokat a `data/` mappába
3. Aggregálja egyetlen CSV-be és generál EDA riportokat

**Manuális alternatíva**: Ha a SharePoint link nem elérhető, helyezd a JSON fájlokat közvetlenül a `data/` mappába.

### Docker Instructions

Ez a projekt teljes mértékben Dockerben fut. Az alábbi parancsok segítségével build-elheted és futtathatod a megoldást.

#### Build

Futtasd az alábbi parancsot a repository gyökérkönyvtárában a Docker image build-eléséhez:

```bash
docker build -t dl-project-legal-text-decoder .
```

#### Run - Training Only (log mentéssel)

Csak a tréning pipeline futtatásához (adat feldolgozás, tréning, értékelés) használd az alábbi parancsot. **A log mentéséhez (beadáshoz szükséges) átirányítjuk a kimenetet egy fájlba:**

Windows PowerShell-ben:
```powershell
docker run --rm --gpus all `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  dl-project-legal-text-decoder > log/run.log 2>&1
```

- A `> log/run.log 2>&1` rész biztosítja, hogy minden kimenet (stdout és stderr) mentésre kerül a `log/run.log` fájlba
- A konténer végigfuttatja az összes lépést: adat előkészítés, tréning, értékelés

#### Run - Full Stack (API + Frontend)

API és frontend szolgáltatások indításához használd az alábbi parancsot:

Windows PowerShell-ben:
```powershell
docker run --rm --gpus all `
  -e START_API_SERVICE=1 `
  -e START_FRONTEND_SERVICE=1 `
  -e API_URL=http://localhost:8000 `
  -p 8000:8000 `
  -p 8501:8501 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  dl-project-legal-text-decoder > log/run.log 2>&1
```

A szolgáltatások elérhetősége:
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8501

#### Fejlesztői műhely - Local futtatás (src/run.sh)

A projekt **Nvidia GPU-n** lett fejlesztve. Local gépen történő futtatáshoz a `src/run.sh` script-et kell futtatni:

**Előfeltételek:**
- Python 3.10+
- Nvidia GPU és CUDA 12.9 telepítve
- Python virtual environment aktiválva

**Adat- és output könyvtárak:**

A projekt fejlesztésére az alábbi könyvtárak lesznek alapértelmezésben használatosak:
- **Data mappa**: `C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data`
- **Output mappa**: `C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output`

### File Structure and Functions

A repository az alábbi struktúrát követi:

- **`src/`**: A gépi tanulási pipeline forráskódja
    - `01_data_acquisition_and_analysis.py`: Automatikus adatletöltés SharePoint-ról, JSON feldolgozás, EDA metrikák és vizualizációk
    - `02_data_cleansing_and_preparation.py`: Szöveg tisztítás, deduplikáció, stratifikált split (train/val/test)
    - `03_baseline_model.py`: HuBERT baseline modell tréning overfitting sanity check-kel, test értékelés (confusion matrix, metrika ábrák)
    - `04_incremental_model_development.py`: Progresszív modellfejlesztés (4 architektúra), automatikus összehasonlítás, legjobb modell test set értékelése (confusion matrix, classwise/avg/error metrika ábrák)
    - `05_defining_evaluation_criteria.py`: Teszt értékelés részletes ábrákkal (confusion matrix, metrics summary, classwise precision/recall/F1, average metrics, error metrics)
    - `06_advanced_evaluation.py`: Robusztusság tesztelés (perturbációs ábrák) és magyarázhatóság (confusion pairs, attention analysis)
    - `07_start_api_service.py`: FastAPI backend indítása
    - `08_start_frontend_service.py`: Streamlit frontend indítása
    - `utils.py`: Logger konfigurációs segédfüggvények
    - `run.sh`: Teljes pipeline végrehajtó bash script
    - **`api/`**: FastAPI backend alkalmazás
        - `app.py`: REST API végpontok (predict, health, models)
    - **`frontend/`**: Streamlit frontend alkalmazás
        - `app.py`: Interaktív webes felület osztály valószínűségekkel

- **`notebook/`**: Jupyter notebookok elemzéshez és kísérletezéshez
    - `01_data_acquisition_and_analysis.ipynb`: Exploratív adatelemzés és vizualizáció
    - `02_data_cleansing_and_preparation.ipynb`: Adattisztítási folyamat notebook változata
    - `03_baseline_model.ipynb`: Baseline modell kísérletezés
    - `04_incremental_model_development.ipynb`: Progresszív modellfejlesztés notebook
    - `05_defining_evaluation_criteria.ipynb`: Értékelési szempontok definiálása
    - `06_advanced_evaluation.ipynb`: Haladó értékelési módszerek
    - `07_08_services.ipynb`: API és frontend szolgáltatások tesztelése
    - `teszteles.ipynb`: Ad-hoc tesztelési notebook

- **`log/`**: Log fájlok könyvtára
    - `run.log`: Sikeres tréning futás kimenetét tartalmazó példa log fájl

- **Gyökérkönyvtár**:
    - `Dockerfile`: Docker image konfigurációs fájl a szükséges környezettel és függőségekkel
    - `requirements.txt`: Python függőségek listája pontos verziószámokkal
    - `README.md`: Projekt dokumentáció és használati útmutató
    - `payload.json`: Példa API request payload
    - `LICENSE`: Projekt licensz fájl

### Output Structure

- `output/raw/`: Aggregált nyers CSV és EDA vizualizációk
- `output/processed/`: `train.csv`, `val.csv`, `test.csv`
- `output/models/`: Model checkpointok (`best_*.pt`), label mapping JSON
- `output/reports/`: Riportok, confusion matrix-ok, összehasonlító grafikonok
- `output/advanced/`: Robusztusság és magyarázhatóság eredményei
- `output/training_log.txt`: Összevont tréning log (`run.sh` kimenet)

### Logging

A tréning folyamat átfogó log fájlt készít az alábbi információkkal:

1. **Konfiguráció**: Hyperparaméterek (epoch-ok száma, batch méret, learning rate, stb.)
2. **Adat feldolgozás**: Sikeres adat betöltés és előfeldolgozási lépések megerősítése
3. **Modell architektúra**: Modell struktúra összefoglalása paraméterszámokkal (trainable/non-trainable)
4. **Tréning folyamat**: Loss és accuracy (vagy más metrikák) logolása minden epoch-nál
5. **Validáció**: Validációs metrikák logolása minden epoch végén
6. **Végső értékelés**: Teszt eredmények (accuracy, MAE, F1-score, confusion matrix)

A log fájl a `log/run.log` fájlba kerül a Docker output átirányításával (`> log/run.log 2>&1`). A logok könnyen érthetőek és önmagyarázóak.

### Dependencies

A projekt függőségei a `requirements.txt` fájlban találhatók pontos verziószámokkal:

**Core:**
- numpy, pandas, scikit-learn, matplotlib, seaborn

**ML stack:**
- torch>=2.0.0, transformers>=4.35.0, sentence-transformers, textstat

**API & Frontend:**
- fastapi, uvicorn, pydantic, streamlit, plotly

**Dev:**
- jupyter, python-dotenv