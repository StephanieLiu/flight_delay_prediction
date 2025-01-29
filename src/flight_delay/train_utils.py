from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer,LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from feature_engine.creation import CyclicalFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import logging
import mlflow
import os,sys
# from mlflow.models.signature import infer_signature

# Get the current directory path
current_directory = os.getcwd()

# Get the grandparent directory
grandparent_dir = os.path.abspath(os.path.join(current_directory, '..', 'src'))

# Add the parent directory to sys.path
parent_dir = os.path.join(grandparent_dir, 'flight_delay')
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'helpers'))
import utils

logger = logging.getLogger('train_utils')


def mark_outliers_by_category(df, category_column, value_column):
    outlier_flag = df.groupby(category_column)[value_column].transform(lambda x: (x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) | (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
    return outlier_flag

#function to create the preprocessing pipeline
def create_preprocess_pipeline(categorical_cols, numerical_cols, time_cols, passthrough_cols):
    # Pipeline for categorical features
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Pipeline for numerical features
    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
       
    # Pipeline to directly pass through features
    passthrough_transformer = Pipeline([
        ('identity', FunctionTransformer(validate=False))
    ])
    
    # Create column transformer
    pre_processor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_cols),
            ('categorical', categorical_transformer, categorical_cols),
            ('cyclical', CyclicalFeatures(drop_original=True), time_cols),
            ('passthrough', passthrough_transformer, passthrough_cols)
        ],
        remainder='drop'  # or 'passthrough' if you want to keep columns that are not specified
    )
    
    return pre_processor

def train_test_split_func(df,train_start,train_end,test_start,test_end,split_col,target_col):
    
    df_train = df[(df[split_col] >= train_start) & (df[split_col] < train_end)].reset_index(drop=True)
    y_train = df_train[target_col]
    X_train = df_train.drop(target_col,axis = 1)
    
    df_test = df[(df[split_col] >= test_start) & (df[split_col] < test_end)].reset_index(drop=True)
    y_test = df_test[target_col]
    X_test = df_test.drop(target_col,axis = 1)
    
    return X_train,y_train,X_test,y_test

def train_model(regressor, pre_processor, X_train, y_train):
    model_pipeline = Pipeline([
        ("pre_processor", pre_processor),
        ("regressor", regressor)
    ])
    model = model_pipeline.fit(X_train,y_train)
    return model

        
def evaluate_model(model,X_train, y_train,X_test,y_test):
    y_train_predictions = model.predict(X_train)
    MAE_train = mean_absolute_error(y_train,y_train_predictions)

    y_test_predictions = model.predict(X_test)
    MAE_test = mean_absolute_error(y_test,y_test_predictions)

    metrics = {
        "MAE_train" : MAE_train,
        "MAE_test" : MAE_test
    }

    return metrics

def train_model_with_tracking(experiment,regressor,regressor_name, X_train, y_train,X_test,y_test,categorical_cols, numerical_cols, time_cols, passthrough_cols):
    
    results = []
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    logger.info('Training and evaluation for regressor: %s', regressor)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"run_{regressor_name}_{utils.truncated_uuid4()}"):
        mlflow.set_tag("regressor model", regressor_name)
        try:
            model_pipeline = Pipeline([
            ("pre_processor", create_preprocess_pipeline(categorical_cols, numerical_cols, time_cols, passthrough_cols)),
            ("regressor", regressor)
        ])
            model_pipeline.fit(X_train,y_train)
            logger.info('Training completed for regressor: %s', regressor)
        except Exception:
            logger.exception('Error training regressor %s', regressor)

        try:
            evaluation_metrics = evaluate_model(model_pipeline,X_train, y_train,X_test,y_test)
            results.append({'regressor': regressor_name, **evaluation_metrics})
            logger.info(
                'Evaluation completed for regressor: %s with metrics: %s',
                regressor,
                evaluation_metrics,
            )
            mlflow.log_metric("train MAE", evaluation_metrics['MAE_train'])
            mlflow.log_metric("test MAE", evaluation_metrics['MAE_test'])

            sample = X_train.sample(5)
            # signature = infer_signature(sample, model_pipeline.predict(sample))
            mlflow.sklearn.log_model(
                    model_pipeline,
                    artifact_path="regressor",
                    input_example=sample,
                    # signature=signature,
                    extra_pip_requirements=["flight_delay"],
                )
        except Exception:
            logger.exception('Error evaluating regressor %s', regressor_name)

    return results