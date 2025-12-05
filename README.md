# Deep Learning Class (VITMMA19) – Legal Text Decoder

Magyar jogi szövegek (ÁSZF/ÁFF) érthetőségének automatikus értékelése 1–5 skálán, modern NLP-vel (HuBERT + olvashatósági feature fusion), REST API + Streamlit GUI, Dockerrel csomagolva.

## Project Information
- **Téma**: Legal Text Decoder
- **Hallgató**: Nagypál Márton Péter
- **Cél**: Outstanding Level (+1 mark)

## Megoldás röviden
- **Modell**: HuBERT alapú transformer, opcionális FusionModel (olvashatósági feature-ök) és CORAL (ordinal) loss; baseline: könnyített transformer finomhangolás.
- **Pipeline lépések**: 01 adatbegyűjtés+EDA, 02 tisztítás+split, 03 baseline tréning/értékelés, 04 fejlett modell (fusion/CORAL), 05 értékelés, 06 advanced (robusztusság + magyarázhatóság), 08 API, 09 Frontend.
- **Szolgáltatás**: FastAPI (8000) + Streamlit GUI (8501), opcionálisan indítható környezeti változókkal.

## Outstanding követelmények – megfelelés
- Containerization ✅ (Dockerfile, opcionális compose)
- Data acquisition & analysis ✅ (`src/01_data_acquisition_and_analysis.py`)
- Data cleansing & preparation ✅ (minden tisztítás most 02-ben)
- Defining evaluation criteria ✅ (05 – transformer értékelés)
- Baseline model ✅ (03 – baseline transformer)
- Incremental model development ✅ (04 – fusion + CORAL)
- Advanced evaluation ✅ (06 – robustness + explainability egyben)
- ML as a service ✅ (08 API, 09 Frontend)

## Adatelőkészítés (teljesen automatizált)
1. **Nyers adat**: `data/` alatt JSON-ek (Label Studio formátum: `data.text`, `annotations`).
2. **01 – begyűjtés + EDA (nem módosít)**: `src/01_data_acquisition_and_analysis.py` → `raw/raw_dataset.csv` + EDA ábrák/statok. Nincs tisztítás vagy duplikált törlés itt.
3. **02 – tisztítás + split**: `src/02_data_cleansing_and_preparation.py`
     - Kisbetűsítés, Unicode normalizálás, whitespace összecsukás, speciális karakter-szűrés.
     - Üres text/label sorok elhagyása; duplikációk törlése kisbetűsített szövegen.
     - Kimenet: `processed/train.csv`, `val.csv`, `test.csv` + szövegstatisztikák.
4. **Futtatás**: mindez automatikusan lefut a `run.sh`-ban (lásd alább).

## Logging
- A `src/run.sh` minden kimenetet az `OUTPUT_DIR` alatti `training_log.txt`-be tükröz (tee). Docker futtatásnál irányítsd a konténer STDOUT-ját `log/run.log`-ba a beadáshoz.
- Tartalmaz: hyperparaméterek, adatbetöltés, modell architektúra, epóchonkénti loss/acc/F1, validáció, végső teszt metrikák (MAE, RMSE, F1, confusion matrix).

## Docker használat (compose nélkül is működik)

### Build
```bash
docker build -t deeplearning_project-legal_text_decoder:1.0 .
```

### Csak training (alap)
```powershell
docker run --rm --gpus all `
    -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
    -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
    deeplearning_project-legal_text_decoder:1.0 `
    > log/run.log 2>&1
```
Linux/macOS:
```bash
docker run --rm --gpus all \
    -v "$(pwd)/attach_folders/data:/app/data" \
    -v "$(pwd)/attach_folders/output:/app/output" \
    deeplearning_project-legal_text_decoder:1.0 \
    > log/run.log 2>&1
```

### Training + API
```powershell
docker run --rm --gpus all `
    -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
    -p 8000:8000 `
    -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
    -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
    deeplearning_project-legal_text_decoder:1.0 `
    > log/run.log 2>&1
```

### Training + API + Frontend
```powershell
docker run --rm --gpus all `
    -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
    -e START_FRONTEND_SERVICE=1 -e API_URL=http://localhost:8000 -e FRONTEND_HOST=0.0.0.0 -e FRONTEND_PORT=8501 `
    -p 8000:8000 -p 8501:8501 `
    -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
    -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
    deeplearning_project-legal_text_decoder:1.0 `
    > log/run.log 2>&1
```
(Linux/macOS: cseréld a volume path-okat `$(pwd)`-re.)

### Compose (opcionális)
Ha mégis compose-t használnál: `docker-compose up training-with-api` vagy `training-full-stack`.

## Fájlstruktúra (fő elemek)
- `src/01_data_acquisition_and_analysis.py` – nyers adat begyűjtés + EDA (nem tisztít)
- `src/02_data_cleansing_and_preparation.py` – tisztítás, kisbetűs dedup, split
- `src/03_baseline_model.py` – baseline transformer tréning + riportok
- `src/04_incremental_model_development.py` – fejlett modell (fusion, CORAL)
- `src/05_defining_evaluation_criteria.py` – értékelés a transformerre
- `src/06_advanced_evaluation.py` – robusztusság + magyarázhatóság (egyesítve)
- `src/08_start_api_service.py` – FastAPI indítás
- `src/09_start_frontend_service.py` – Streamlit indítás
- `src/api/app.py` – REST API
- `src/frontend/app.py` – GUI
- `src/run.sh` – teljes pipeline futtatása (01–06, opcionálisan 08–09)
- `notebook/teszteles.ipynb` – kísérletek
- `attach_folders/data`, `attach_folders/output` – host-oldali mount pontok

## Eredmények / Kimenetek
- `output/models/` – baseline és best transformer checkpointok, label mapping
- `output/reports/` – lépésenkénti riportok, ábrák (EDA, baseline, advanced)
- `output/training_log.txt` – futási log (tee). Beadáskor másold `log/run.log`-ba vagy irányítsd oda a konténer kimenetét.

## Extra Credit (indoklás)
- FusionModel (transformer + olvashatósági feature-ök), CORAL loss (ordinal), robusztussági vizsgálat (zaj/csonkítás), attention-alapú magyarázhatóság, teljes szolgáltatáslánc (API + GUI) Dockerben.
