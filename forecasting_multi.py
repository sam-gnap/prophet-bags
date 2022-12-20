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
from scalecast import GridGenerator

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
def pre_process_bags(df):
    df = df.loc[df["lid"] > 100]
    df = df.loc[df["sold"] > 20]
    df["bags"] = df["23kgbags"] + df["15kgbags"]
    df["bags_conversions"] = df["bags"] / df["sold"]
    df = df[["sector", "std", "bags_conversions"]]
    df.loc[df["bags_conversions"] > 1, "bags_conversions"] = 1
    df.columns = ["sector", "ds", "y"]
    return df
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

# COMMAND ----------

orynce = df.loc[
    df["sector_x"] == "ORYNCE", ["localdepdt", "rps", "conversions"]
].reset_index(drop=True)

nceory =  df.loc[
    df["sector_x"] == "NCEORY", ["localdepdt", "rps", "conversions"]
].reset_index(drop=True)

# COMMAND ----------

forynce = Forecaster(y=orynce.rps, current_dates = orynce.localdepdt, future_dates = 14, freq='d')
fnceory = Forecaster(y=nceory.rps, current_dates = nceory.localdepdt, future_dates = 14, freq='d')

# COMMAND ----------

for f in (forynce,fnceory):
    f.set_test_length(.05) # all models will be tested out of sample
    f.add_time_trend()
    f.add_seasonal_regressors('month','quarter','week','dayofyear',raw=False,sincos=True)
    f.add_seasonal_regressors('dayofweek','week',raw=False,dummy=True,drop_first=True)
    f.add_seasonal_regressors('year')
    f.add_ar_terms(365)

# COMMAND ----------

# download template validation grids (will not overwrite existing Grids.py file by default)
models = ('mlr','elasticnet','knn','rf','gbt','xgboost','mlp', 'prophet')
#GridGenerator.get_example_grids()
#GridGenerator.get_mv_grids()

# COMMAND ----------

forynce.tune_test_forecast(models,feature_importance=True)
forynce.set_estimator('combo')
forynce.manual_forecast(how='weighted')

# COMMAND ----------

fnceory.tune_test_forecast(models,feature_importance=True)
fnceory.set_estimator('combo')
fnceory.manual_forecast(how='weighted')

# COMMAND ----------

forynce.plot_test_set(ci=False, include_train=False,order_by='LevelTestSetMAPE',models=['xgboost','prophet'])
plt.title('Conventional Univariate Test-set Results',size=16)
plt.show()

# COMMAND ----------

fnceory.plot_test_set(ci=False, include_train=False,order_by='LevelTestSetMAPE',models=['xgboost','prophet'])
plt.title('Conventional Univariate Test-set Results',size=16)
plt.show()

# COMMAND ----------

pd.set_option('display.float_format',  '{:.4f}'.format)
ms = export_model_summaries({'orynce':forynce, 'nceory':fnceory},determine_best_by='LevelTestSetMAPE')
ms[
    [
        'ModelNickname',
        'Series',
        'Integration',
        'LevelTestSetMAPE',
        'LevelTestSetR2',
        'InSampleMAPE',
        'InSampleR2',
        'best_model'
    ]
]

# COMMAND ----------

# MAGIC %md
# MAGIC # Multivariate

# COMMAND ----------

# download template validation grids (will not overwrite existing Grids.py file by default)
models = ('mlr','ridge','knn','rf','gbt','xgboost','mlp', 'prophet')
GridGenerator.get_example_grids()
GridGenerator.get_mv_grids()

# COMMAND ----------

mvf = MVForecaster(forynce,fnceory,names=['orynce','nceory']) # init the mvf object
mvf.set_test_length(.05)
mvf.set_validation_length(8)
mvf

# COMMAND ----------

mvf.corr()

# COMMAND ----------

mvf.set_optimize_on('mean')

# COMMAND ----------

mvf.tune_test_forecast(models)

# COMMAND ----------


mvf.set_best_model(determine_best_by='LevelTestSetMAPE')

# COMMAND ----------


