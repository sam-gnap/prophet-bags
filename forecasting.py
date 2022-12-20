# Databricks notebook source
import pandas as pd
import itertools
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

pd.set_option("display.max_rows", 100)

import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scalecast.MVForecaster import MVForecaster
from scalecast.multiseries import export_model_summaries
from scalecast.SeriesTransformer import SeriesTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC The best result so far has changepoint_prior_scale of 0.001 seasonality_prior_scale of 0.01 with seasonality_mode being multiplicative 15.456195 RMSE.

# COMMAND ----------

def prepare_data(df_events):
    df_events["ds"] = pd.to_datetime(
        (
            df_events["StartDate"] + (df_events["EndDate"] - df_events["StartDate"]) / 2
        ).dt.date
    )
    df_events["lower_window"] = (df_events["StartDate"] - df_events["ds"]).dt.days
    df_events["upper_window"] = np.abs((df_events["EndDate"] - df_events["ds"])).dt.days
    df_events = df_events.loc[
        (df_events["Origin"] == "LGW") | (df_events["Destination"] == "AMS"),
        ["Name", "ds", "lower_window", "upper_window"],
    ].reset_index(drop=True)
    df_events.columns = ["holiday", "ds", "lower_window", "upper_window"]
    lockdowns = pd.DataFrame(
        [
            {
                "holiday": "lockdown_1",
                "ds": "2020-03-21",
                "lower_window": 0,
                "ds_upper": "2021-05-01",
            },
        ]
    )
    for t_col in ["ds", "ds_upper"]:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns["upper_window"] = (lockdowns["ds_upper"] - lockdowns["ds"]).dt.days
    df_events = pd.concat(
        [df_events, lockdowns[["holiday", "ds", "lower_window", "upper_window"]]]
    ).reset_index(drop=True)
    return df_events

# COMMAND ----------

def pre_process_bags(df):
    df = df.loc[df["lid"] > 100]
    df = df.loc[df["sold"] > 20]
    df["bags"] = df["23kgbags"] + df["15kgbags"]
    df["bags_conversions"] = df["bags"] / df["sold"]
    df = df[["sector", "std", "bags_conversions"]]
    df.loc[df["bags_conversions"] > 1, "bags_conversions"] = 1
    df.columns = ["sector", "ds", "y"]
    return df


rps = spark.read.parquet(
    "dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet"
).toPandas()
df_bags = spark.read.parquet(
    "dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet"
).toPandas()
df = pd.merge(df_bags, rps, right_on="flightkey", left_on="segment")
df["rps"] = df["ticketrevenuegbpbudgetnet"] / df["lid_y"]
df["yield"] = df["ticketrevenuegbpbudgetnet"] / df["seatssold"]
df["conversions_23"] = df["23kgbags"] / df["seatssold"]
df["conversions_15"] = df["15kgbags"] / df["seatssold"]
df["conversions"] = (df["23kgbags"] + df["15kgbags"]) / df["seatssold"]
df_bags = pre_process_bags(df_bags)
orynce = df_bags.loc[df_bags["sector"] == "ORYNCE"]
df = df[
    [
        "sector_x",
        "localdepdt",
        "seatssold",
        "rps",
        "yield",
        "conversions_23",
        "conversions_15",
        "conversions",
    ]
]
df = df.loc[df["seatssold"] > 100]
df = df.groupby(["sector_x", "localdepdt"]).agg(
    {
        "seatssold": np.sum,
        "rps": np.mean,
        "yield": np.mean,
        "conversions_23": np.mean,
        "conversions_15": np.mean,
        "conversions": np.mean,
    }
)
df = df.reset_index(drop=False)
orynce = df.loc[
    df["sector_x"] == "ORYNCE", ["localdepdt", "rps", "conversions"]
].reset_index(drop=True)

# COMMAND ----------

df.head()

# COMMAND ----------

df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()

# COMMAND ----------

def process_holidays(events, sector,start,end):
    events = events.loc[(events['Origin']==sector[:3])|(events['Destination']==sector[3:]),
           ['Name','StartDate','EndDate','Origin','Destination']].sort_values(by=['Name','StartDate']).reset_index(drop=True)
    lockdowns = pd.DataFrame([
    {'Name': 'lockdown', 'StartDate': '2020-03-21', 'EndDate': '2021-05-01'},])
    events = pd.concat([events,lockdowns]).reset_index(drop=True)
    events['StartDate'] = pd.to_datetime(events['StartDate'])
    events['EndDate'] = pd.to_datetime(events['EndDate'])
    d = events['EndDate'].sub(events['StartDate']).dt.days
    df1 = events.reindex(events.index.repeat(d))
    i = df1.groupby(level=0).cumcount() + 1

    df1['date'] = df1['StartDate'] + pd.to_timedelta(i, unit='d')
    df1 = df1[['Name','date']].drop_duplicates()
    df1['value'] = 1
    df1 = pd.pivot_table(df1,
                  index='date',
                  columns='Name',
                  values='value').reset_index()
    df1 = df1.replace(np.nan,0)
    idx = pd.date_range(start, end)
    df1.index = pd.DatetimeIndex(df1.date)
    df1 = df1.reindex(idx, fill_value=0)
    df1 = df1.reset_index()
    df1 = df1.drop(columns=['date'])
    df1.columns.name=None
    df1 = df1.rename(columns={ df1.columns[0]: "Date" })
    return df1

# COMMAND ----------

holidays_df = process_holidays(df_events, sector='ORYNCE',start='2017-01-01',end='2023-06-01')

# COMMAND ----------

events = prepare_data(df_events)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Scalecast
# MAGIC ### Set-up

# COMMAND ----------

orynce

# COMMAND ----------

f = Forecaster(
    y = orynce.rps,
    current_dates = orynce.localdepdt,
    future_dates = 14,
    freq='d'
)

# COMMAND ----------

from scalecast.auxmodels import mlp_stack, auto_arima

# COMMAND ----------

models = (
    'knn',
    'mlr',
    'elasticnet',
    'ridge',
    'sgd',
    'xgboost',
    'lightgbm',
    'mlp',
    'arima',
    'prophet',
)

# COMMAND ----------

f.add_ar_terms(1)
f.add_AR_terms((3,6))
f.add_AR_terms((3,12))
auto_arima(
    f,
    m=12,
    error_action='ignore',
)
f.manual_forecast(
    order=f.auto_arima_params['order'],
    seasonal_order=f.auto_arima_params['seasonal_order'],
    Xvars = 'all',
    call_me='auto_arima_anom',
)
f.tune_test_forecast(
    models,
    limit_grid_size=10,
    cross_validate=True,
    k=3,
    dynamic_tuning=fcst_horizon,
    suffix='_cv_uv',
)
f.tune_test_forecast(
    models,
    limit_grid_size=10,
    cross_validate=False,
    dynamic_tuning=fcst_horizon,
    suffix='_tune_uv',
)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

f.set_test_length(.05) # all models will be tested out of sample
f.add_time_trend()

f.add_seasonal_regressors('month','quarter','week','dayofyear',raw=False,sincos=True)
f.add_seasonal_regressors('dayofweek','week',raw=False,dummy=True,drop_first=True)
f.add_seasonal_regressors('year')
f.add_ar_terms(365)

# COMMAND ----------

models = (
    'knn',
    'mlr',
    'elasticnet',
    'sgd',
    'xgboost',
    'lightgbm',
    'mlp',
    'hwes',
    'arima',
    'prophet',
    'silverkite',
    'theta',
)


# COMMAND ----------

f.set_estimator('mlr')
f.manual_forecast(dynamic_testing=14)

# COMMAND ----------

f.set_estimator('prophet')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

f.set_estimator('lasso')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

f.set_estimator('ridge')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

f.set_estimator('elasticnet')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

f.set_estimator('sgd')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

f.set_estimator('lightgbm')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

f.plot_test_set(ci=False, include_train=False,models=['ridge','prophet'])
plt.show()

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------

!pip install datetime

# COMMAND ----------

import datetime

# COMMAND ----------

f.add_covid19_regressor(called='COVID19', start=datetime.datetime(2020, 3, 15, 0, 0), end=datetime.datetime(2021, 5, 13, 0, 0))

# COMMAND ----------

f.set_estimator('ridge')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

f.ingest_Xvars_df(holidays_df,date_col="Date")

# COMMAND ----------

f

# COMMAND ----------

f.set_estimator('prophet')
f.cross_validate(k=5)
f.auto_forecast()

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------

f.plot_test_set(ci=True, include_train=False,models=['ridge'])
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

f.manual_forecast(how='weighted',determine_best_by=None,models=['ridge','prophet'],weights=(1,2),call_me='manual_weighted_avg')

# COMMAND ----------

f.manual_forecast(how='weighted',models=['ridge','prophet'],determine_best_by='TestSetMAPE',call_me='weighted_avg_insampler2')

# COMMAND ----------

f.manual_forecast(how='weighted',models=['ridge','prophet'],determine_best_by='InSampleMAPE',call_me='weighted_avg_insampler3')

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------



# COMMAND ----------

f.ingest_Xvars_df(holidays_df,date_col="Date")

# COMMAND ----------

f.set_estimator('arima')
f.manual_forecast(order=(3,1,3),seasonal_order=(2,1,2,12))

# COMMAND ----------

f.set_estimator('combo')
f.manual_forecast(call_me='simple_avg')

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

f.manual_forecast(how='weighted',determine_best_by=None,models=['ridge','lasso'],weights=(2,1),call_me='manual_weighted_avg')

# COMMAND ----------

f.manual_forecast(how='weighted',models=['ridge','sgd'],determine_best_by='InSampleMAPE',call_me='weighted_avg_insampler2')

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------

f.plot_test_set(ci=False, include_train=False, models=['weighted_avg_insampler2'])
plt.title('Manual Weighted Average Test Results',size=16)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Adding transformations

# COMMAND ----------

from scalecast.Forecaster import Forecaster
from scalecast.Pipeline import Pipeline, Transformer, Reverter
from scalecast.auxmodels import mlp_stack
from scalecast import GridGenerator
import matplotlib.pyplot as plt
import pandas_datareader as pdr

# COMMAND ----------

models = (
  'mlr',
  'elasticnet',
  'lightgbm',
  'knn',
)

GridGenerator.get_example_grids()

f = Forecaster(
  y=orynce['rps'],
  current_dates=orynce['localdepdt'],
  future_dates=90,
)
f.set_test_length(.05) # all models will be tested out of sample
f.add_time_trend()

f.add_seasonal_regressors('month','quarter','week','dayofyear',raw=False,sincos=True)
f.add_seasonal_regressors('dayofweek','week',raw=False,dummy=True,drop_first=True)
f.add_seasonal_regressors('year')
f.add_ar_terms(365)
f.ingest_Xvars_df(holidays_df,date_col="Date")

def forecaster(f,models):
    """ add Xvars and forecast
    """
    f.auto_Xvar_select()
    f.tune_test_forecast(
        models,
        dynamic_testing=24, # test-set metrics will be an average of rolling 24-step forecasts
        cross_validate=True,
        k = 3,
    )
    mlp_stack(f,models)
        
transformer = Transformer(
    transformers = [
        ('DiffTransform',1),
        ('DiffTransform',12),
    ],
)
reverter = Reverter(
    # list reverters in reverse order
    reverters = [
        ('DiffRevert',12),
        ('DiffRevert',1),
    ],
    base_transformer = transformer,
)
pipeline = Pipeline(
    steps = [
        ('Transform',transformer),
        ('Forecast',forecaster),
        ('Revert',reverter),
    ],
)

f = pipeline.fit_predict(f,models=models)

f.reeval_cis() # expanding cis based on all model results

# COMMAND ----------

f.plot(ci=True,order_by='LevelTestSetMAPE')
plt.show()

# COMMAND ----------

f.plot_test_set(include_train=False)
plt.show()

# COMMAND ----------

results = f.export(
  ['model_summaries','lvl_fcsts']
)

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------

d = f.export(dfs=['model_summaries','lvl_fcsts','test_set_predictions'],to_excel=False, best_model='mlp_stack')

# COMMAND ----------

d.keys()

# COMMAND ----------

d['test_set_predictions'].head(20)

# COMMAND ----------


