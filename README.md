# Deep Learning Class (VITMMA19) – Legal Text Decoder

## Projekt összefoglaló (valós pipeline)

- **Téma**: Jogi szövegek érthetőségének automatikus értékelése (1–5 skála).
- **Fő megközelítés**: HuBERT-alapú transzformátor finomhangolása osztályozásra.
- **Progresszív modellek**: 4 variáns lépésenként növekvő kapacitással (Baseline, Extended, Advanced, Final Balanced) a transzformer tetején adapterekkel/gatinggel. Jelenlegi pipeline nem használ külön olvashatósági feature-fúziót és nem alkalmaz CORAL ordinal loss-t.

## Pipeline lépések

1. Adatgyűjtés és EDA – [src/01_data_acquisition_and_analysis.py](src/01_data_acquisition_and_analysis.py)
   - Label Studio formátumú JSON-ok feldolgozása; EDA metrikák és vizualizációk mentése.
2. Tisztítás és előkészítés – [src/02_data_cleansing_and_preparation.py](src/02_data_cleansing_and_preparation.py)
   - Normalizálás, deduplikáció, 60/20/20 arányú, osztály-arányos split; szövegstatisztikák mentése.
3. Baseline tréning – [src/03_baseline_model.py](src/03_baseline_model.py)
   - HuBERT-alapú transzformer, CrossEntropy és standard riportok (accuracy, F1, MAE/RMSE az ordinal skálához).
4. Progresszív modellek – [src/04_incremental_model_development.py](src/04_incremental_model_development.py)
   - 4 architektúra variáns edzése; a legjobb checkpoint mentése `best_<modell>.pt` formában.
5. Értékelés – [src/05_defining_evaluation_criteria.py](src/05_defining_evaluation_criteria.py)
   - Teszt riportok és konfúziós mátrix. Megjegyzés: a kód jelen állapotában csak transzformer-alapú inference-et használ.
6. Haladó értékelés – [src/06_advanced_evaluation.py](src/06_advanced_evaluation.py)
   - Robusztusság (zaj, csonkítás) és attention-alapú magyarázhatóság. A modul a transzformer checkpointot tölti be.
7. API – [src/07_start_api_service.py](src/07_start_api_service.py), [src/api/app.py](src/api/app.py)
   - FastAPI szolgáltatás. A jelenlegi tréning kimenethez igazítani szükséges a betöltési útvonalakat/checkpoint formátumot.
8. Frontend – [src/08_start_frontend_service.py](src/08_start_frontend_service.py), [src/frontend/app.py](src/frontend/app.py)
   - Streamlit GUI, az API-ra támaszkodik.

## Futtatás Dockerben

### Build

```bash
docker build -t deeplearning_project-legal_text_decoder:1.0 .
```

### Training only

```powershell
docker run --rm --gpus all `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > log/run.log 2>&1
```

### Teljes stack (API + Frontend)

Megjegyzés: A jelenlegi tréning kimenet `.pt` checkpointokat hoz létre a `output/models/` könyvtárban. Az API betöltése ehhez igazításra szorul (modellkönyvtár/név egyeztetés). A konténer indítása:

```powershell
docker run --rm --gpus all `
  -e START_API_SERVICE=1 -e API_HOST=0.0.0.0 -e API_PORT=8000 `
  -e START_FRONTEND_SERVICE=1 -e API_URL=http://localhost:8000 -e FRONTEND_HOST=0.0.0.0 -e FRONTEND_PORT=8501 `
  -p 8000:8000 -p 8501:8501 `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\data:/app/data" `
  -v "C:\Users\nagyp\.vscode\DeepLearning Project\attach_folders\output:/app/output" `
  deeplearning_project-legal_text_decoder:1.0 > log/run.log 2>&1
```

## Kimeneti struktúra

- `output/raw/`: aggregált nyers CSV és EDA ábrák
- `output/processed/`: `train.csv`, `val.csv`, `test.csv`
- `output/models/`: `best_<modell>.pt` (PyTorch checkpointok), `label_mapping.json`
- `output/reports/`: baseline és progresszív modellek riportjai/ábrái
- `output/training_log.txt`: összevont napló a `run.sh` futásról

## Követelmények és környezet

- **CUDA**: A Docker image `pytorch/pytorch:2.8.0-cuda12.9` (GPU támogatás).
- **Adatforrás**: Label Studio JSON; a data mappa a konténerben `/app/data`.
- **Konfiguráció**: Környezeti változók (pl. `OUTPUT_DIR`, `START_API_SERVICE`, `START_FRONTEND_SERVICE`).

## Ismert eltérések / teendők

- Az API jelenleg egy `best_transformer_model` könyvtárat vár; a tréning `.pt` fájlokat hoz létre. A betöltés egységesítése szükséges.
- A README korábbi verziója FusionModel + CORAL-t említett. A kód jelenleg tisztán transzformer + CrossEntropy megközelítést valósít meg.
- A `requirements.txt` frissítve lett és tisztítva (lásd alább).

## Függőségek

A projekt egységesített függőségei a `requirements.txt`-ben találhatók (duplikációk eltávolítva, verziók összehangolva).