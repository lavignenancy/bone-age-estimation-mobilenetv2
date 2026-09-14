# Bone Age Estimation — Coursework Prototype

**AkiraChix DAS — Medical Imaging ML Project**

> ⚠️ **This is a student coursework prototype, not a medical device.** It has not been clinically validated or
> reviewed by a radiologist. Never use it to make a real decision about a real person's health.

## Project purpose

Predicts bone age (in months) from a single left-hand X-ray image, as a regression task, using the RSNA
Pediatric Bone Age dataset. Built as coursework to demonstrate an end-to-end ML pipeline: data inspection,
preprocessing, model training, honest evaluation, and a working Streamlit portal.

## Dataset source

- **RSNA Pediatric Bone Age Challenge (2017)**, Kaggle mirror: `https://www.kaggle.com/datasets/kmader/rsna-bone-age`
- Requires a free Kaggle account to download.
- **Do not** confuse this with `kwankhaotangprasert/rsna-handmask-create-masked-bone-age-images` — that is a
  separate *preprocessing notebook*, not the dataset itself, and is not the primary source for this project.
- Check the licence/usage terms on the Kaggle dataset page yourself before any use beyond coursework.

## Expected file structure

```
bone_age_project/
├── data/                              # your downloaded kmader/rsna-bone-age files (not included in this repo)
├── bone_age_project.ipynb             # Task 0 + sections A1-A8: inspection, training, evaluation
├── model_utils.py                     # SHARED preprocessing/model code (imported by both notebook and app)
├── app.py                             # Streamlit app
├── requirements.txt
├── README.md
└── models/                            # created after you run the notebook's A7 export section
    ├── bone_age_model.keras
    └── preprocessing_config.json
```

`model_utils.py` must stay in the **same folder** as both `bone_age_project.ipynb` and `app.py` — both import it
directly.

## How to install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## How to run things, in order

1. **Inspect and train** — open `bone_age_project.ipynb` in Jupyter Notebook, set `DATA_ROOT` to your dataset
   folder in the Setup cell, and run every cell top to bottom. This produces `models/bone_age_model.keras` and
   `models/preprocessing_config.json`.
2. **Run the app** — from the project's root folder (the one containing `app.py` and `models/`):
   ```bash
   streamlit run app.py
   ```
3. Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Accepted image formats

PNG and JPG/JPEG (matches the formats found in the RSNA/kmader dataset — confirmed during notebook Section A2).

## Model file location

The app expects the trained model at `models/bone_age_model.keras` and its config at
`models/preprocessing_config.json`, both relative to wherever you run `streamlit run app.py` from. If you move
the model, either move it back or update `MODEL_DIR` at the top of `app.py`.

## Preprocessing

Every image (in training, evaluation, and the app) is: converted to RGB, resized to 224x224, then scaled using
`tf.keras.applications.mobilenet_v2.preprocess_input`. This logic lives in exactly one place —
`model_utils.load_and_preprocess_image()` — imported by both the notebook and the app, so they can never
silently drift apart.

## Output meaning

The app outputs a single number: estimated bone age in months (plus a years/months conversion for readability).
It is a statistical estimate based on learned patterns from the training dataset, not a certified clinical
reading, and comes with a typical error (MAE, in months) reported in the notebook's evaluation section (A6) and
shown in the app under "How reliable is this number?".

## Limitations (see notebook section A8 for full detail)

- Trained and evaluated only on the RSNA/kmader dataset — no external validation on a different hospital or
  population.
- No sex/gender input — image-only model, by design (see A4).
- No calibrated confidence interval — only an average historical error (MAE) is reported, not a per-prediction
  confidence.
- Patient-level duplication across splits could not be fully ruled out — see A2/A3 for what was and wasn't
  checked.

## Medical disclaimer

This project is a machine learning coursework exercise. It is **not** a diagnostic tool, has **not** been
reviewed or approved by any medical or regulatory body, and must **never** be used to make decisions about a
real person's health or treatment. If you have genuine questions about bone age, growth, or development, consult
a qualified healthcare professional.
