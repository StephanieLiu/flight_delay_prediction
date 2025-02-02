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

from typing import List, Dict
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run
from flight_delay.mlflow.tracking import build_model_uri_from_run

def get_all_model_versions(registry_uri:str, model_name:str) -> List[Dict]:
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    filter_string = f"name='{model_name}'"
    results = client.search_model_versions(filter_string)
    return [{"version": mv.version, "run_id":mv.run_id, "aliases": mv.aliases}
            for mv in results]

def get_latest_model_versions(registry_uri:str, model_name:str) -> List[Dict]:
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    filter_string = f"name='{model_name}'"
    results = client.search_registered_models(filter_string=filter_string)
    latest_versions = []
    for res in results:
        latest_versions += [{"version": mv.version, "run_id":mv.run_id, "aliases": mv.aliases}
                            for mv in res.latest_versions]
    return latest_versions

def set_active_production_model(registry_uri:str, 
                                production_model_name:str, model_version:str) -> None:
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    client.set_registered_model_alias(production_model_name, "active", model_version)

def set_backup_production_model(registry_uri:str, 
                                production_model_name:str, model_version:str) -> None:
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    client.set_registered_model_alias(production_model_name, "backup", model_version)

def transition_model_to_staging(registry_uri:str, 
                                model_name:str, model_version:str) -> None:
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    client.set_registered_model_alias(model_name, "staging", model_version)

def promote_model_to_production(registry_uri:str, 
                                model_name:str, 
                                production_model_name:str,
                                model_alias:str="staging") -> None:
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    return client.copy_model_version(
                src_model_uri=build_registered_model_uri(model_name, alias=model_alias), 
                dst_name=production_model_name,
            )
    
def update_model_description(registry_uri:str, model_name:str, 
                             model_version:str, description:str) -> None:
    """
    Updates the description of a specific model version.

    Args:
    - registry_uri (str): The URI where the MLflow model registry server is running.
    - model_name (str): The name of the model to update.
    - model_version (str): The version of the model to update.
    - description (str): The new description for the model version.
    """
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=description
    )
        
def tag_model(registry_uri:str, 
              model_name:str, model_version:str, tags:Dict) -> None:
    """
    Tags a specific model version with provided key-value pairs.

    Args:
    - registry_uri (str): The URI where the MLflow registry server is running.
    - model_name (str): The name of the model to tag.
    - model_version (str): The version of the model to tag.
    - tags (Dict): A dictionary containing key-value pairs to tag the model version.
    """
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    for tag_k, tag_v in tags.items():
        # Tag using model version
        client.set_model_version_tag(name=model_name, 
                                    version=f'{model_version}', 
                                    key=tag_k, value=tag_v)
    
def load_model(registry_uri:str, 
               model_name:str, 
               model_version:str=None,
               model_alias:str=None) -> Pipeline:
    mlflow.set_tracking_uri(registry_uri)
    model_uri = build_registered_model_uri(model_name, model_version, model_alias)
    loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)
    return loaded_model

def get_version_of_model_by_alias(registry_uri, model_name, model_alias):
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    model_version = client.get_model_version_by_alias(model_name, model_alias)
    return model_version.version

def download_registered_model(registry_uri, model_name, model_version, local_dst_dirpath):
    mlflow.set_tracking_uri(registry_uri)
    client = MlflowClient(tracking_uri=registry_uri)
    model_artifact_uri = client.get_model_version_download_uri(model_name, model_version)
    mlflow.artifacts.download_artifacts(model_artifact_uri, dst_path=local_dst_dirpath)

def register_model_from_run(tracking_uri:str, run:Run, model_name:str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = build_model_uri_from_run(run)
    return mlflow.register_model(model_uri=model_uri, 
                                name=model_name)

def build_registered_model_uri(name:str, version:str=None, alias:str=None):
    assert version != None or alias != None, "either version or alias should be given!"
    if version:
        model_uri = f"models:/{name}/{version}"
    else:
        model_uri = f"models:/{name}@{alias}"
    return model_uri


