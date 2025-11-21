# Content Monetization Modeler 💰

A Machine Learning project to predict YouTube ad revenue based on video engagement metrics.

## Project Structure

- **`src/`**: Modular Python scripts for data processing and training.
    - `preprocess.py`: Feature engineering and Scikit-Learn pipeline.
    - `train.py`: Script to train the XGBoost model.
- **`app/`**: Streamlit web application.
    - `app.py`: The main application file.
- **`Data/`**: Contains the dataset (`USvideos.csv`).
- **`Models/`**: Stores the trained model (`ad_revenue_model.joblib`).
- **`Notebooks/`**: Jupyter notebooks for exploration.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model
To retrain the model using the latest data:
```bash
python src/train.py
```
This will load `Data/USvideos.csv`, clean it (remove duplicates), train an XGBoost model, and save it to `Models/ad_revenue_model.joblib`.

### 2. Run the App
To launch the web interface:
```bash
streamlit run app/app.py
```
Open the URL provided in the terminal (usually `http://localhost:8501`) to use the app.
