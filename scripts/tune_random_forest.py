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
from dotenv import load_dotenv

load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_URI")
import warnings
warnings.filterwarnings("ignore")
import argparse
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp
from hyperopt.pyll import scope
from bank_marketing.models.skl_tracked_tune_models import tune_with_tracking 
from bank_marketing.data.prep_datasets import prepare_binary_classfication_tabular_data
from bank_marketing.data.make_datasets import make_bank_marketing_dataframe
from bank_marketing.features.skl_build_features import FeatureNames
from bank_marketing.models.skl_tracked_train_models import Experiment
from bank_marketing.helpers.file_loaders import load_yaml_file
from bank_marketing.data.load_datasets import load_dataset_from_dvc


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding arguments
    parser.add_argument("-e", "--experiment_name", 
                        default='tune_random_forest',
                        help="MLFlow Experiment Name")
    parser.add_argument("-d", "--data", 
                        default="data.yml",
                        help="Bank Marketing Data Card")
    parser.add_argument("-m", "--max_runs", 
                        default=10,
                        help="Maximum run number")
    parser.add_argument('--data-dir', type=str, 
                        default='../data', 
                        help='Path to data directory')
    
    # Read arguments from command line
    args = parser.parse_args()
    experiment = Experiment(MLFLOW_URI, args.experiment_name)
    ds_info = load_yaml_file(args.data)

    dataset, dvc_info = load_dataset_from_dvc(data_dir=args.data_dir)
    ds_info['dvc_info'] = dvc_info

    feature_names = FeatureNames(ds_info["numerical_features"], ds_info["categorical_features"]) 
    hparams =  {
        "max_depth":scope.int(hp.quniform("max_depth", 1, 30, q=1)),
        "max_features":hp.uniform("max_features", 0.05, 0.8),
        "class_weight":hp.choice("class_weight", ["balanced", None]),
        "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf", 5, 100, q=5))
    }

    tune_with_tracking(
        dataset,
        feature_names,
        RandomForestClassifier,
        hparams,
        int(args.max_runs),
        experiment,
        ds_info
    )
    