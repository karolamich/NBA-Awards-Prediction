# NBA Regular Season Awards Prediction

> **Note:** This project was developed as part of a Machine Learning course assignment.

## Introduction
This project utilizes machine learning techniques to predict which players will be nominated for the NBA regular season awards. The system generates rankings and selects the rosters for two groups:
* **All-NBA Team** (three teams - 15 players)
* **All-Rookie Team** (two teams - 10 rookies)

The final predictions were generated for the 2025-26 season.

## Technologies & Requirements
* **Language:** Python 3
* **Key Libraries (ML & Data):** CatBoost, XGBoost, LightGBM, Random Forest, `nba_api`, `unidecode` (along with other dependencies defined in `requirements.txt`)
* **Data Sources:** NBA API and web scraping from Basketball Reference (covering 26 seasons: 2000-01 to 2025-26).

## Project Structure

**Note:** The repository contains only the source code. The datasets and trained models are generated locally during the execution of the scripts.

* `files/` – Scripts and Jupyter notebooks containing the project code:
  * `1_scraping.ipynb` - Data collection
  * `2_preprocessing.ipynb` - Initial data preparation
  * `2_preprocessing_expanded.ipynb` - Expanded data preparation (team stats & games started)
  * `3_model.ipynb` - Baseline model evaluation
  * `3_model_allnba.ipynb` - Feature engineering and model exploration for All-NBA
  * `3_model_final.ipynb` - Final CatBoost models training and evaluation
  * `4_make_json.py` - Prediction script
* `.gitignore` – Git ignored files list.
* `requirements.txt` – List of environment dependencies.

*(The `data/` and `models/` directories are intentionally excluded from version control and will be created automatically when running the project pipeline).*

## Installation and Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the data processing pipeline (in an environment that supports `.ipynb` notebooks):
   * **Step 1:** Download data from external sources.
     Run `files/1_scraping.ipynb` (this will automatically create the `data/raw/` directory and save downloaded data).
   * **Step 2:** Data preparation and expansion.
     Run `files/2_preprocessing_expanded.ipynb` (this will create `data/processed/` and process the data).
   * **Step 3:** Final models training.
     Run `files/3_model_final.ipynb` (this will create the `models/` directory, save the trained `.pkl` models, and prepare final data in `data/final/`).

## Usage (Generating Predictions)

Before running the prediction script, ensure you have executed the setup steps to generate the models and final datasets. Then, run the main Python script from the command line, providing an absolute path to the output JSON file.

```bash
python3 files/4_make_json.py /absolute/path/to/output.json
```
The script loads the final models from the `models/` directory, applies the necessary filters to the test set (e.g., minimum of 63 games played for the All-NBA group), and saves the final rankings to the specified `.json` file.

## Model Technical Details
* **Algorithm:** The project uses the **CatBoostRegressor** model optimized for predicting voting results (percentage share of maximum voting points).
* **Hyperparameters (All-NBA):** `depth=6`, `iterations=200`, `learning_rate=0.05`
* **Hyperparameters (All-Rookie):** `depth=5`, `iterations=300`, `learning_rate=0.1`
