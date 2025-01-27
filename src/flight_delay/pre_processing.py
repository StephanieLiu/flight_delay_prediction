import pandas as pd
import numpy as np

def load_data(data_path):
    df = pd.read_csv(data_path,encoding="utf-8")
    return df

def cast_columns_types(df_flights):
    df_flights['FL_DATE'] = pd.to_datetime(df_flights['FL_DATE'])
    df_flights['AIRLINE'] = df_flights['AIRLINE'].astype('str')
    df_flights['AIRLINE_DOT'] = df_flights['AIRLINE_DOT'].astype('str')
    df_flights['AIRLINE_CODE'] = df_flights['AIRLINE_CODE'].astype('str')
    df_flights['ORIGIN'] = df_flights['ORIGIN'].astype('str')
    df_flights['ORIGIN_CITY'] = df_flights['ORIGIN_CITY'].astype('str')
    df_flights['DEST'] = df_flights['DEST'].astype('str')
    df_flights['DEST_CITY'] = df_flights['DEST_CITY'].astype('str')
    df_flights["schd_dep_hour"] = df_flights["CRS_DEP_TIME"].astype("str")
    return df_flights

def create_nulls_stat(df):
    missing_value_stat = df.isnull().sum()
    df_missing_value_stat = pd.DataFrame(missing_value_stat).reset_index()
    df_missing_value_stat.columns = ['column_name','nulls_number']
    df_missing_value_stat['null_percentage'] = round(df_missing_value_stat['nulls_number'] / df.shape[0] * 100, 2)
    return df_missing_value_stat

def pre_process(df):
    
    df= cast_columns_types(df)
    
    #create covid flag
    df["covid_data"] = np.where((df["FL_DATE"] >= "2020-01-01") & (df["FL_DATE"] < "2022-01-01"), 1, 0)
    
    #drop canceled and diverted flights
    df = df[(df.CANCELLED == 0) & (df.DIVERTED== 0)]
    
    #drop rows with null in target variable
    df = df.dropna(subset=["ARR_DELAY"])

    #drop unnecessary columns
    df.drop(columns=["AIRLINE_DOT",'DOT_CODE',"FL_NUMBER","CANCELLED",'CANCELLATION_CODE',"DIVERTED","DELAY_DUE_CARRIER","DELAY_DUE_WEATHER","DELAY_DUE_NAS","DELAY_DUE_SECURITY","DELAY_DUE_LATE_AIRCRAFT"],inplace=True)
    
    #extract seasonal features
    df = extract_seasonal_features(df)
    
    return df

def extract_seasonal_features(df):
    df["flight_year"] = df["FL_DATE"].dt.year
    df["flight_month"] = df["FL_DATE"].dt.month
    df["flight_day_of_month"] = df["FL_DATE"].dt.day
    df["flight_day_of_week"] = df["FL_DATE"].dt.dayofweek
    df["schd_dep_hour"] = df["CRS_DEP_TIME"].astype("str").str[:-2]
    df.loc[df["schd_dep_hour"]=="", "schd_dep_hour"] = "0"
    return df

# def mark_outliers_by_category(df, category_column, value_column):
#     outlier_flag = df.groupby(category_column)[value_column].transform(lambda x: (x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) | (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
#     return outlier_flag