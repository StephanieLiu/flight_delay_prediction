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

SERVER_API_URL = os.environ.get('SERVER_API_URL')
FUTURE_RES_DF = os.environ.get('FUTURE_RES_DF')
FSSPEC_S3_KEY = os.environ.get('FSSPEC_S3_KEY')
FSSPEC_S3_SECRET = os.environ.get('FSSPEC_S3_SECRET')
FSSPEC_S3_ENDPOINT_URL = os.environ.get('FSSPEC_S3_ENDPOINT_URL')

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
st.write("Debug Information:")
st.write(f"Data path: {FUTURE_RES_DF}")
st.write(f"MinIO endpoint: {FSSPEC_S3_ENDPOINT_URL}")
st.write(f"Access key exists: {'Yes' if FSSPEC_S3_KEY else 'No'}")
st.write(f"Secret key exists: {'Yes' if FSSPEC_S3_SECRET else 'No'}")

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
        st.write("Available buckets:", fs.ls(''))
        
        # Get bucket and path
        bucket = data_path.split('/')[2]
        path = '/'.join(data_path.split('/')[3:])
        
        # List contents of bucket
        st.write(f"Contents of bucket '{bucket}':", fs.ls(bucket))
        
        # Full path for debugging
        full_path = f"{bucket}/{path}"
        st.write(f"Attempting to open: {full_path}")
        
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

def check_model_endpoint(url: str) -> bool:
    """
    Check if the model endpoint is available and responding.
    Returns True if the model is available, False otherwise.
    """
    try:
        # Remove '/infer' from the URL to get the model endpoint
        model_url = url.replace('/infer', '')
        
        # Make a GET request to the model metadata endpoint
        response = requests.get(
            model_url,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            metadata = response.json()
            st.write("Debug: Model metadata:", metadata)
            return True
        else:
            st.error(f"Model endpoint returned status code: {response.status_code}")
            st.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        st.error("Model endpoint check timed out")
        return False
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to model endpoint")
        return False
    except Exception as e:
        st.error(f"Error checking model endpoint: {str(e)}")
        return False

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


    st.write("Debug: Available columns:", future_results_data.columns.tolist())
    st.write("Debug: Required columns:")
    st.write("- Numerical:", numerical_cols)
    st.write("- Categorical:", categorical_cols)
    st.write("- Time:", time_cols)
    st.write("- Passthrough:", passthrough_cols)

    # Verify all required columns exist
    missing_cols = []
    for col in numerical_cols + categorical_cols + time_cols + passthrough_cols:
        if col not in future_results_data.columns:
            missing_cols.append(col)
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        try:
            with st.spinner('Transforming data...'):
                future_results_data_sample = future_results_data.sample(10)

                future_results_data_transformed = train_utils.predict_with_pipeline(
                    future_results_data_sample,
                    categorical_cols, 
                    numerical_cols, 
                    time_cols, 
                    passthrough_cols
                )
                
                # Display transformed data
                st.subheader("First 10 Rows of transformed Flight Data")
                st.dataframe(future_results_data_transformed.head(10), use_container_width=True)
                
                # Show transformed data shape
                st.write(f"Transformed data shape: {future_results_data_transformed.shape}")

                # Add this after loading the data and before making predictions
                st.write("Checking model endpoint...")
                if not check_model_endpoint(SERVER_API_URL):
                    st.error("Model is not available. Please check the model deployment.")
                    st.stop()
                else:
                    st.success("Model endpoint is available!")

                # Make predictions using the model
                st.write("Debug: Server URL:", SERVER_API_URL)
                
                try:
                    # Prepare data for prediction
                    prediction_data = future_results_data_transformed.to_dict(orient='records')
                    
                    # Debug request information
                    st.write("Debug: Making prediction request")
                    st.write("URL:", SERVER_API_URL)
                    st.write("Request data shape:", len(prediction_data))
                    st.write("Sample request data:", prediction_data[0])
                    
                    # Make prediction request
                    response = requests.post(
                        SERVER_API_URL,
                        json={"inputs": prediction_data},
                        headers={"Content-Type": "application/json"},
                        timeout=30  # Add timeout
                    )
                    st.write("Debug: Response status code:", response.status_code)
                    st.write("Debug: Response headers:", dict(response.headers))
                    
                    if response.status_code == 200:
                        predictions = response.json()
                        st.success("Predictions received successfully!")
                        
                        # Add predictions to the original data
                        future_results_data_sample['predicted_delay'] = predictions['predictions']
                        
                        # Display results
                        st.subheader("Predictions")
                        st.dataframe(
                            future_results_data_sample[['OP_CARRIER', 'ORIGIN', 'DEST', 'predicted_delay']]
                            .head(10), 
                            use_container_width=True
                        )
                    else:
                        st.error(f"Error from server: {response.status_code}")
                        st.error(f"Response: {response.text}")
                    
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The server took too long to respond.")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the server. Please check if the server is running.")
                except Exception as e:
                    st.error(f"Error making prediction request: {str(e)}")
                    st.error(f"Full error details: {repr(e)}")

        except Exception as e:
            st.error(f"Error during transformation: {str(e)}")
            st.error(f"Full error details: {repr(e)}")

except Exception as e:
    st.error(f"Error loading flight data: {str(e)}")


