import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from preprocess import get_pipeline
import os

def train_model():
    # Load Data
    # Use path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../Data/USvideos.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Data Cleaning
    print(f"Original shape: {df.shape}")
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")
    
    # Define Target and Features
    # Dropping target and potential leakage columns if they exist in source
    X = df.drop(columns=['ad_revenue_usd', 'video_id', 'date', 'trending_date', 'trending_time', 'tags', 'thumbnail_link', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'description'], errors='ignore')
    
    # Keep only relevant columns for X
    # Note: FeatureEngineer will handle engagement_rate creation
    relevant_cols = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'category', 'device', 'country']
    X = X[relevant_cols]
    
    y = df['ad_revenue_usd']

    # Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get Pipeline
    preprocessing_pipeline = get_pipeline()
    
    # Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    
    # Full Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline),
        ('regressor', model)
    ])

    # Train
    print("Training model...")
    full_pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = full_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # Save Model
    # Use path relative to this script
    model_dir = os.path.join(script_dir, '../Models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, 'ad_revenue_model.joblib')
    joblib.dump(full_pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
