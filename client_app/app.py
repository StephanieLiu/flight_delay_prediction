# Copyright (c) 2024 Houssem Ben Braiek, Emilio Rivera-Landos, IVADO, SEMLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import streamlit as st
import fsspec
import requests
import json


current_directory = os.getcwd()

# Get the grandparent directory
grandparent_dir = os.path.abspath(os.path.join(current_directory, '..', 'src'))
sys.path.append(grandparent_dir)

from flight_delay import train_utils,features
from flight_delay.mlflow.registry import load_model

SERVER_API_URL = os.environ.get('SERVER_API_URL')
FUTURE_RES_DF = os.environ.get('FUTURE_RES_DF')
FSSPEC_S3_KEY = os.environ.get('FSSPEC_S3_KEY')
FSSPEC_S3_SECRET = os.environ.get('FSSPEC_S3_SECRET')
FSSPEC_S3_ENDPOINT_URL = os.environ.get('FSSPEC_S3_ENDPOINT_URL')

# Add MLflow environment variables
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.environ.get('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.environ.get('MLFLOW_TRACKING_PASSWORD')

@st.cache_resource
def connect_to_data(data_path):
    try:
        fs = fsspec.filesystem(
            's3',
            key=FSSPEC_S3_KEY,
            secret=FSSPEC_S3_SECRET,
            endpoint_url=FSSPEC_S3_ENDPOINT_URL,
            client_kwargs={'endpoint_url': FSSPEC_S3_ENDPOINT_URL}
        )
        
        # List buckets to verify connection
        #st.write("Available buckets:", fs.ls(''))
        
        # Get bucket and path
        bucket = data_path.split('/')[2]
        path = '/'.join(data_path.split('/')[3:])
        
        # List contents of bucket
        #st.write(f"Contents of bucket '{bucket}':", fs.ls(bucket))
        
        # Full path for debugging
        full_path = f"{bucket}/{path}"
        #st.write(f"Attempting to open: {full_path}")
        
        # Read data in chunks with a progress bar
        with st.spinner('Loading data... This might take a while...'):
            chunks = []
            with fs.open(full_path) as f:
                # Read in chunks of 100,000 rows
                for chunk in pd.read_csv(f, chunksize=100000):
                    chunks.append(chunk)
            
            # Combine all chunks
            data = pd.concat(chunks, ignore_index=True)
            
        return data
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        st.error(f"Full error details: {repr(e)}")
        raise


def get_model_predictions(data: pd.DataFrame):
    """
    Get predictions using model loaded directly from MLflow
    """
    try:
        # Load the model from MLflow registry
        model = load_model(
            registry_uri=MLFLOW_TRACKING_URI,
            model_name="lightGBM-production",  # Use your model name
            model_alias="active"  # Or specific version if needed
        )
        
        # Make predictions
        predictions = model.predict(data)
        return predictions
        
    except Exception as e:
        st.error(f"Error loading model from MLflow: {str(e)}")
        st.error(f"Full error details: {repr(e)}")
        raise

# --- APP SETTINGS ---
page_title = 'Flight Delay Prediction'
page_icon = '✈️'
layout = 'wide'

# --------- Page SETUP -----------
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .title-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0066cc;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Title section with description
with st.container():
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f'<h1 style="font-size: 4rem">{page_icon}</h1>', unsafe_allow_html=True)
    with col2:
        st.title('Flight Delay Prediction')
        st.markdown('Predict flight delays using machine learning')
    st.markdown('</div>', unsafe_allow_html=True)

# Data loading section
with st.container():
    try:
        # Debug information
        st.write("Getting data from:")
        st.write(f"Data path: {FUTURE_RES_DF}")
        st.write(f"MinIO endpoint: {FSSPEC_S3_ENDPOINT_URL}")
        # st.write(f"Access key exists: {'Yes' if FSSPEC_S3_KEY else 'No'}")
        # st.write(f"Secret key exists: {'Yes' if FSSPEC_S3_SECRET else 'No'}")
        future_results_data = connect_to_data(FUTURE_RES_DF)
        
        # Display metrics in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(future_results_data):,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Flights</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{future_results_data["AIRLINE"].nunique()}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Airlines</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{future_results_data["ORIGIN"].nunique()}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Airports</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Sample data display
        with st.expander("View Sample Data", expanded=False):
            st.dataframe(future_results_data.head(10), use_container_width=True)

        # Debug information for columns
        numerical_cols = features.numerical_cols
        categorical_cols = features.categorical_cols
        time_cols = features.time_cols
        passthrough_cols = features.passthrough_cols
        features_list = numerical_cols + categorical_cols + time_cols + passthrough_cols


        # st.write("Debug: Available columns:", future_results_data.columns.tolist())
        # st.write("Debug: Required columns:")
        # st.write("- Numerical:", numerical_cols)
        # st.write("- Categorical:", categorical_cols)
        # st.write("- Time:", time_cols)
        # st.write("- Passthrough:", passthrough_cols)

        # Verify all required columns exist
        missing_cols = []
        for col in numerical_cols + categorical_cols + time_cols + passthrough_cols:
            if col not in future_results_data.columns:
                missing_cols.append(col)
            
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")

        # Predictions section
        if not missing_cols:
            with st.container():
                with st.spinner('Making predictions...'):
                    future_results_data_sample = future_results_data.sample(1000)
                    predictions = get_model_predictions(future_results_data_sample)
                    future_results_data_sample['predicted_delay'] = predictions

                # Results in tabs
                tab1, tab2 = st.tabs(["📊 Delay Analysis", "📋 Detailed Results"])
                
                with tab1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.subheader("Average Predicted Delay by Airline")
                    
                    # Calculate and display average delays
                    avg_delay_by_airline = (
                        future_results_data_sample
                        .groupby('AIRLINE')['predicted_delay']
                        .agg(['mean', 'count'])
                        .round(2)
                        .reset_index()
                    )
                    avg_delay_by_airline.columns = ['Airline', 'Average Delay (minutes)', 'Number of Flights']
                    avg_delay_by_airline = avg_delay_by_airline.sort_values('Average Delay (minutes)', ascending=True)
                    
                    # Bar chart
                    st.bar_chart(
                        data=avg_delay_by_airline.set_index('Airline')['Average Delay (minutes)'],
                        use_container_width=True
                    )

                    # Add new visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.subheader("Top 10 Origin Cities with Highest Delays")
                        origin_delays = (
                            future_results_data_sample
                            .groupby('ORIGIN_CITY')['predicted_delay']
                            .agg(['mean', 'count'])
                            .round(2)
                            .sort_values('mean', ascending=False)
                            .head(10)
                            .reset_index()
                        )
                        origin_delays.columns = ['Origin City', 'Average Delay (minutes)', 'Number of Flights']
                        
                        st.bar_chart(
                            data=origin_delays.set_index('Origin City')['Average Delay (minutes)'],
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.subheader("Top 10 Destination Cities with Highest Delays")
                        dest_delays = (
                            future_results_data_sample
                            .groupby('DEST_CITY')['predicted_delay']
                            .agg(['mean', 'count'])
                            .round(2)
                            .sort_values('mean', ascending=False)
                            .head(10)
                            .reset_index()
                        )
                        dest_delays.columns = ['Destination City', 'Average Delay (minutes)', 'Number of Flights']
                        
                        st.bar_chart(
                            data=dest_delays.set_index('Destination City')['Average Delay (minutes)'],
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Add temporal analysis
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.subheader("Average Delay by Date")
                    
                    # Convert FL_DATE to datetime if it's not already
                    future_results_data_sample['FL_DATE'] = pd.to_datetime(future_results_data_sample['FL_DATE'])
                    
                    # Daily average delays
                    daily_delays = (
                        future_results_data_sample
                        .groupby('FL_DATE')['predicted_delay']
                        .agg(['mean', 'count'])
                        .round(2)
                        .reset_index()
                    )
                    daily_delays.columns = ['Date', 'Average Delay (minutes)', 'Number of Flights']
                    
                    st.line_chart(
                        data=daily_delays.set_index('Date')['Average Delay (minutes)'],
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Add route analysis
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.subheader("Top 10 Routes with Highest Delays")
                    
                    # Create route column
                    future_results_data_sample['Route'] = (
                        future_results_data_sample['ORIGIN_CITY'] + ' → ' + 
                        future_results_data_sample['DEST_CITY']
                    )
                    
                    route_delays = (
                        future_results_data_sample
                        .groupby('Route')['predicted_delay']
                        .agg(['mean', 'count'])
                        .round(2)
                        .sort_values('mean', ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    route_delays.columns = ['Route', 'Average Delay (minutes)', 'Number of Flights']
                    
                    st.bar_chart(
                        data=route_delays.set_index('Route')['Average Delay (minutes)'],
                        use_container_width=True
                    )
                    
                    # Display route details in a table
                    st.markdown("#### Route Details")
                    st.dataframe(
                        route_delays,
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("Detailed Statistics by Airline")
                    st.dataframe(avg_delay_by_airline, use_container_width=True)
                    
                    st.subheader("Sample Predictions")
                    st.dataframe(
                        future_results_data_sample.head(20), 
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error(f"Details: {repr(e)}")






