import sys
import os

# Add src directory to path so pickle can find 'preprocess' module
# This must be done before loading the model
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '../src')
sys.path.append(src_dir)

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Content Monetization Modeler", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '../Models/ad_revenue_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

# Title and Description
st.title("💰 Content Monetization Modeler")
st.markdown("Predict the estimated ad revenue for your YouTube videos based on engagement metrics.")

if model is None:
    st.error("Model not found! Please train the model first by running `python src/train.py`.")
else:
    # Input Form
    with st.form("prediction_form"):
        st.subheader("Video Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            views = st.number_input("Views", min_value=0, value=10000)
            likes = st.number_input("Likes", min_value=0, value=500)
            comments = st.number_input("Comments", min_value=0, value=50)
            
        with col2:
            watch_time = st.number_input("Watch Time (minutes)", min_value=0.0, value=5000.0)
            video_length = st.number_input("Video Length (minutes)", min_value=0.0, value=10.0)
            subscribers = st.number_input("Subscribers", min_value=0, value=1000)
            
        with col3:
            category = st.selectbox("Category", ['Entertainment', 'Gaming', 'Education', 'Tech', 'Music', 'News', 'Sports', 'Comedy', 'Travel', 'Howto'])
            device = st.selectbox("Device", ['Mobile', 'Desktop', 'Tablet', 'TV'])
            country = st.selectbox("Country", ['US', 'IN', 'UK', 'CA', 'DE'])
            
        submit_button = st.form_submit_button("Predict Revenue")

    if submit_button:
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'views': [views],
            'likes': [likes],
            'comments': [comments],
            'watch_time_minutes': [watch_time],
            'video_length_minutes': [video_length],
            'subscribers': [subscribers],
            'category': [category],
            'device': [device],
            'country': [country]
        })
        
        # Predict
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Ad Revenue: **${prediction:,.2f}**")
            
            # Simple Analysis
            st.markdown("---")
            st.subheader("Analysis")
            
            # Engagement Rate Calculation for display
            engagement_rate = ((likes + comments) / views) * 100 if views > 0 else 0
            st.metric("Engagement Rate", f"{engagement_rate:.2f}%")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & XGBoost")
