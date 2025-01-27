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