import os,sys
from dotenv import load_dotenv

load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_URI")
import warnings
warnings.filterwarnings("ignore")
import argparse

from flight_delay import pre_processing,train_utils,features
from flight_delay.mlflow.registry import load_model
from flight_delay.mlflow.tracking import Experiment
import flight_delay.mlflow.registry as registry
import flight_delay.mlflow.tracking as tracking
from lightgbm import LGBMRegressor

current_directory = os.getcwd()

# Get the grandparent directory
grandparent_dir = os.path.abspath(os.path.join(current_directory, '..', 'src'))
sys.path.append(grandparent_dir)

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train_parser")
    parser.add_argument("-e", "--experiment_name", 
                        default='retrain_flt_delay_model',
                        help="MLFlow Experiment Name")
    parser.add_argument("--train_start_date", type=str, 
                        default='2019-01-01',
                        help="train_start_date")
    parser.add_argument("--train_end_date", type=str, 
                        default='2023-03-01',
                        help="train_end_date")
    parser.add_argument('--data-dir', type=str, 
                        default='../data/flights_delay_dataset.csv', 
                        help='Path to data directory')
    arguments = parser.parse_args()
    return arguments

def main(args):
    data_path = '/home/ivamlops/flight_delay_prediction/data/flights_delay_dataset.csv'
    df_flights = pre_processing.load_data(data_path)
    df = df_flights[df_flights["covid_data"] == 0]

    target_col = "ARR_DELAY"
    split_col  = "FL_DATE"

    test_end = df[split_col].max()
    train_start = args.train_start_date
    train_end = args.train_end_date
    try:
        X_train,y_train,X_test,y_test = train_utils.train_test_split_func(df,train_start,train_end,train_end,test_end,split_col,target_col)
        print("--------------------------------------")
        print("Train and test data are generated.")
    except:
        print("Got error when creating train and test data.")


    numerical_cols = features.numerical_cols
    categorical_cols = features.categorical_cols
    time_cols = features.time_cols
    passthrough_cols = features.passthrough_cols

    features_list = numerical_cols + categorical_cols + time_cols + passthrough_cols

    predicted_var = "predicted_arrival_delay"

    print("--------------------------------------")
    print("Retrain model with new data.")
    experiment_name = 'lightGBM'
    experiment = Experiment(MLFLOW_URI, experiment_name)
    results_new = train_utils.train_model_with_tracking(experiment,LGBMRegressor(),LGBMRegressor.__name__, X_train, y_train,X_test,y_test,categorical_cols, numerical_cols, time_cols, passthrough_cols)
    
    best_run = tracking.get_best_run(experiment)
    model_name, model_version = "lightGBM", "2"
    model_under_test = load_model(MLFLOW_URI, model_name, model_version)
    results_current = train_utils.evaluate_model(model_under_test,X_train, y_train,X_test,y_test)

    if results_new[0]["MAE_test"]<results_current["MAE_test"]:
        model_name = "lightGBM"
        model_version = registry.register_model_from_run(experiment.tracking_server_uri, best_run, model_name)
        registry.transition_model_to_staging(experiment.tracking_server_uri, model_name, model_version.version)
        model_version = registry.promote_model_to_production(experiment.tracking_server_uri, model_name, f"{model_name}-production")
        registry.set_active_production_model(experiment.tracking_server_uri,  f"{model_name}-production", model_version)
        print("--------------------------------------")
        print(f"Successfully registered new model {model_name}:{model_version} in mlflow")
    else:
        print("new model doesn't perform better than current model, won't register the new model.")

if __name__ == "__main__":
    args = parse_args()
    main(args)





