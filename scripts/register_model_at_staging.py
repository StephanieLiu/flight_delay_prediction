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
import os
import logging
import argparse
from dotenv import load_dotenv
load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_URI")
from bank_marketing.mlflow.tracking import explore_best_runs, get_best_run
from bank_marketing.mlflow.tracking import Experiment
import bank_marketing.mlflow.registry as registry

logger = logging.getLogger('scripts.transition_model_to_staging')

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding arguments
    parser.add_argument("-e", "--experiment_name", 
                        default='tune_random_forest_with_kmeans',
                        help="MLFlow Experiment Name")
    parser.add_argument("-m", "--model_name", 
                        default="camp-accept-predictor",
                        help="MLFlow Experiment Name")
    # Read arguments from command line
    args = parser.parse_args()
    experiment = Experiment(MLFLOW_URI, args.experiment_name)

    logger.info("Results of the best models:\n%s", explore_best_runs(experiment).to_string(index=False))

    # Identify the best run
    best_run = get_best_run(experiment)

    # Register the best model using the {model_name} from argument
    model_version = registry.register_model_from_run(experiment.tracking_server_uri, best_run, args.model_name)
    
    # Transition the registered model above to staging
    registry.transition_model_to_staging(experiment.tracking_server_uri, args.model_name, model_version.version)

    
    