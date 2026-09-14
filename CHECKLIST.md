# Testing, Debugging, and Submission Checklist

## Local / Jupyter testing checklist

- [ ] **DATA_ROOT** in the notebook's Setup cell points to your real dataset folder
- [ ] **ID_COLUMN, LABEL_COLUMN, SEX_COLUMN** were checked and are correct (not left as auto-guessed **None**)
- [ ] A2's matched/unmatched counts make sense (unmatched counts near zero, or explained if not)
- [ ] A3's split-overlap check prints **0** for all three pairs (train/val, train/test, val/test)
- [ ] Training actually ran to completion (or early-stopping triggered) without crashing
- [ ] The training/validation loss and MAE curves were plotted and you wrote an honest observation
- [ ] A6's metrics table, plots, and worst/best example images were generated from the real test set
- [ ] **models/bone_age_model.keras** and **models/preprocessing_config.json** exist after running A7
- [ ] The reloaded-model-same-prediction check in A7 printed a near-zero difference

## Streamlit app testing checklist

Test the app with all of these, and note what happened for each:

- [ ] A normal, supported hand X-ray image → produces a specific numeric prediction
- [ ] An unreadable or corrupt file (e.g. rename a **.txt** file to **.png**) → app shows a clear error, does not crash
- [ ] A very small or blank image → app shows the too-small or looks-blank warning, does not attempt a prediction
- [ ] An image from your test set (one the model never trained on) → prediction appears, compare it mentally to the actual label from your notebook
- [ ] An image you know the model performs poorly on (from A6's worst predictions table) → prediction still appears, and you can honestly discuss why it's off in your report

## Consistency checklist

- [ ] The app's **model_utils.py** is the exact same file as the notebook's (not a second, edited copy)
- [ ] A prediction on the same image gives the same result in the notebook (A6/A7) and in the app
- [ ] **requirements.txt** was updated with the exact versions from your own environment (**pip freeze**)
- [ ] The README's install and run instructions work in a fresh virtual environment (test this before submitting)

## Final submission checklist

- [ ] **bone_age_project.ipynb**: fully run, all action-needed write-ups filled in with your real findings
- [ ] **app.py, model_utils.py, requirements.txt**
- [ ] **models/bone_age_model.keras** and **models/preprocessing_config.json** (or a download link plus instructions if the file is too large to submit directly)
- [ ] **README.md**
- [ ] A short note confirming the app runs locally with a fresh **pip install -r requirements.txt**
- [ ] Every markdown section (Task 0, Task 0a, A1–A8) has real, filled-in content — no leftover placeholder text

## Common beginner errors and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| **FileNotFoundError** on **DATA_ROOT** | Path typo, or wrong slashes for your OS | Copy the path directly from your file explorer or terminal; use raw strings (**r"..."**) on Windows |
| **ID_COLUMN / LABEL_COLUMN** printed as **None** | Your CSV uses different column names than the common ones | Set them manually after checking the printed column list |
| Training extremely slow | No GPU, or batch size too high for your machine | Lower **BATCH_SIZE**, or train on a smaller sample subset first while debugging |
| Streamlit app: model file not found | You ran the app from the wrong folder, or have not run A7 yet | **cd** into the project root before **streamlit run app.py**; make sure A7 ran successfully |
| App gives a wildly different prediction than the notebook for the same image | Preprocessing mismatch | Confirm both notebook and app import the same **model_utils.py** file, not two different copies |
| **ValueError** about image shape when predicting | Uploaded image was not converted to RGB or resized before prediction | This should already be handled by **model_utils.load_and_preprocess_image** — if you edited it, check you did not remove the **convert("RGB")** or **resize(...)** steps |
