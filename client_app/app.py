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

# --- APP SETTINGS ---
page_title = 'Flight Delay Prediction'
page_icon = '✈️'
layout = 'wide'

# --------- Page SETUP -----------
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

# Load CSS
with open('styles/main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Flight Delay Prediction')

# Debug information
st.write("Getting data from:")
st.write(f"Data path: {FUTURE_RES_DF}")
st.write(f"MinIO endpoint: {FSSPEC_S3_ENDPOINT_URL}")
# st.write(f"Access key exists: {'Yes' if FSSPEC_S3_KEY else 'No'}")
# st.write(f"Secret key exists: {'Yes' if FSSPEC_S3_SECRET else 'No'}")

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

try:
    future_results_data = connect_to_data(FUTURE_RES_DF)
    
    # Display basic information about the data
    st.success(f"Successfully loaded {len(future_results_data)} rows of flight data!")
    
    # Display first ten rows
    st.subheader("First 10 Rows of Flight Data")
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
    else:
        try:
            with st.spinner('Making predictions...'):
                future_results_data_sample = future_results_data.sample(1000)
                
                # Instead of making API call, use MLflow directly
                predictions = get_model_predictions(future_results_data_sample)
                
                # Add predictions to the original data
                future_results_data_sample['predicted_delay'] = predictions
                
                # Display results
                st.success(f"Successfully made predictions!")

                # Display predictions table
                st.subheader("Sample Predictions")
                st.dataframe(
                    future_results_data_sample.head(20), 
                    use_container_width=True
                )

                # Create and display average delay by airline chart
                st.subheader("Average Predicted Delay by Airline")
                
                # Calculate average delay by airline
                avg_delay_by_airline = (
                    future_results_data_sample
                    .groupby('AIRLINE')['predicted_delay']
                    .agg(['mean', 'count'])
                    .round(2)
                    .reset_index()
                )
                avg_delay_by_airline.columns = ['Airline', 'Average Delay (minutes)', 'Number of Flights']
                
                # Sort by average delay
                avg_delay_by_airline = avg_delay_by_airline.sort_values('Average Delay (minutes)', ascending=True)
                
                # Create bar chart
                st.bar_chart(
                    data=avg_delay_by_airline.set_index('Airline')['Average Delay (minutes)'],
                    use_container_width=True
                )
                
                # Also display the numerical values
                st.write("Detailed Statistics by Airline:")
                st.dataframe(
                    avg_delay_by_airline,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error(f"Full error details: {repr(e)}")

except Exception as e:
    st.error(f"Error loading flight data: {str(e)}")


