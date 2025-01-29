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

from dataclasses import dataclass
from typing import List, Tuple
import json
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run
from sklearn.pipeline import Pipeline

@dataclass
class Experiment:
    """
    A dataclass used to represent an Experiment on MLflow
    Attributes
    ----------
    tracking_server_uri : str
        the URI of MLFlow experiment tracking server
    name : str
        the name of the experiment
    """
    tracking_server_uri:str
    name:str

def load_model_from_run(tracking_server_uri:str, run:Run) -> Pipeline:
    """load the model stored within a given experiment run

    Args:
        tracking_server_uri (str): mlflow tracking server URI
        run (Run): the entity of the given run

    Returns:
        model (Pipeline): the stored model
    """
    mlflow.set_tracking_uri(tracking_server_uri)
    run_model_uri = build_model_uri_from_run(run)
    logged_model = mlflow.sklearn.load_model(run_model_uri)
    return logged_model

def get_best_run(experiment:Experiment, 
                 metric:str="valid_accuracy",
                 order:str="DESC",
                 filter_string:str="") -> Run:
    """Find the best experiment run entity

    Args:
        experiment (Experiment): experiment settings
        metric (str, optional): the metric for runs comparison. Defaults to "valid_accuracy".
        order (str, optional): the sorting order to find the best at first row w.r.t the metric. Defaults to "DESC".
        filter_string (str, optional): a string with which to filter the runs. Defaults to empty string, thus searching all runs.

    Returns:
        Run: the best run entity associated with the given experiment
    """
    best_runs = explore_best_runs(experiment, 1, metric, order, filter_string, False)
    return best_runs[0]

def explore_best_runs(experiment:Experiment, n_runs:int=5, metric:str="valid_accuracy", 
                      order:str="DESC", filter_string:str="", to_dataframe:bool=True) -> List[Run] | pd.DataFrame:
    """find the best runs from the given experiment

    Args:
        experiment (Experiment): Experiment settings
        n_runs (int, optional): the count of runs to return. Defaults to 5.
        metric (str, optional): the metric for runs comparison. Defaults to "valid_accuracy".
        order (str, optional): the sorting order w.r.t the metric to have the best at first row. Defaults to "DESC".
        filter_string (str, optional): a string with which to filter the runs. Defaults to empty string, thus searching all runs.
        to_dataframe (bool, optional): True for a derived Dataframe of Run ID / Perf. Metric. Defaults to True.

    Returns:
        List[Run] | pd.DataFrame: set of the best runs (Entity or Dataframe)
    """
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    client = MlflowClient(tracking_uri=experiment.tracking_server_uri)
    # Retrieve Experiment information
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    # Retrieve Runs information
    runs = client.search_runs(
        experiment_ids=experiment_id,
        max_results=n_runs,
        filter_string=filter_string,
        order_by=[f"metrics.{metric} {order}"]
    )
    if to_dataframe:
        run_ids = [run.info.run_id for run in runs if metric in run.data.metrics]
        run_metrics = [run.data.metrics[metric] for run in runs if metric in run.data.metrics]
        run_dataframe = pd.DataFrame({"Run ID": run_ids, "Perf.": run_metrics})
        return run_dataframe
    return runs

def download_artifact_from_run(tracking_server_uri:str, run:Run, src_fpath: os.PathLike, dest_dirpath: os.PathLike) -> str:
    os.makedirs(dest_dirpath, exist_ok=True)
    client = MlflowClient(tracking_uri=tracking_server_uri)
    run_id = run.info.run_id
    local_artifact_path = client.download_artifacts(run_id, src_fpath, dest_dirpath)
    return local_artifact_path

def build_model_uri_from_run(run:Run) -> str:
    """
    Builds the model URI from the MLflow Run object.

    Args:
    - run (Run): MLflow Run object containing information about the run.

    Returns:
    - str: The model URI constructed from the run information.
    """
    artifact_path = json.loads(run.data.tags['mlflow.log-model.history'])[0]["artifact_path"]
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    return model_uri