# Databricks notebook source
import pandas as pd
import numpy as np
from scalecast.Forecaster import Forecaster
from scalecast.MVForecaster import MVForecaster
from scalecast.SeriesTransformer import SeriesTransformer
from scalecast.AnomalyDetector import AnomalyDetector
from scalecast.ChangepointDetector import ChangepointDetector
from scalecast.util import plot_reduction_errors, break_mv_forecaster, metrics
from scalecast import GridGenerator
from scalecast.multiseries import export_model_summaries
from scalecast.auxmodels import mlp_stack, auto_arima
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm
import pickle

sns.set(rc={'figure.figsize':(12,8)})

# COMMAND ----------

def pre_process_bags(df):
    df = df.loc[df['lid']>100]
    df = df.loc[df['sold']>20]
    df['bags'] = df['23kgbags']+df['15kgbags']
    df['bags_conversions'] = df['bags']/df['sold']
    df = df[['sector','std','bags_conversions']]
    df.loc[df['bags_conversions']>1,'bags_conversions']=1
    df.columns = ['sector','ds','y']
    return df
rps = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()
df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()
df = pd.merge(df_bags,
              rps,
              right_on='flightkey',
              left_on='segment')
df['rps'] = df['ticketrevenuegbpbudgetnet']/df['lid_y']
df['yield'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']
df['conversions_23'] = df['23kgbags']/df['seatssold']
df['conversions_15'] = df['15kgbags']/df['seatssold']
df['conversions'] = (df['23kgbags']+df['15kgbags'])/df['seatssold']
df_bags = pre_process_bags(df_bags)
orynce = df_bags.loc[df_bags['sector']=='ORYNCE']
df = df[['sector_x','localdepdt','seatssold','rps','yield','conversions_23','conversions_15','conversions']]
df = df.loc[df['seatssold']>100]
df = df.groupby(['sector_x', 'localdepdt']).agg({'seatssold': np.sum, 'rps': np.mean, 
                                            'yield': np.mean, 'conversions_23': np.mean, 'conversions_15': np.mean, 
                                            'conversions': np.mean})
df = df.reset_index(drop=False)
orynce = df.loc[df['sector_x']=='ORYNCE',['localdepdt','rps','conversions']].reset_index(drop=True)

# COMMAND ----------

GridGenerator.get_example_grids()

# COMMAND ----------

df.head(2)

# COMMAND ----------

airline_series = ['ORYNCE','NCEORY']
data = df.loc[df['sector_x'].isin(airline_series)]

# COMMAND ----------

fcst_horizon = 90

fdict = {
    sector:Forecaster(
        y=data.loc[data['sector_x']==sector]['rps'],
        current_dates=data.loc[data['sector_x']==sector]['localdepdt'],
        future_dates=fcst_horizon,
    ) for sector in airline_series
}

# COMMAND ----------

sns.set(rc={'figure.figsize':(20,14)})

# COMMAND ----------

for l, f in fdict.items():
    f.plot()
    plt.title(l,size=16)
    plt.show()

# COMMAND ----------

# scan on whole dataset to compare
for l, f in fdict.items():
    tr = SeriesTransformer(f)
    tr.DiffTransform(1)
    tr.DiffTransform(12)
    f2 = tr.ScaleTransform()
    ad = AnomalyDetector(f2)
    ad.MonteCarloDetect_sliding(720,720)
    ad.plot_anom()
    plt.title(f'{l} (diffed series) identified anomalies - all periods',size=16)
    plt.show()

# COMMAND ----------

from scalecast.ChangepointDetector import ChangepointDetector

# COMMAND ----------

for l, f in fdict.items():
    cd = ChangepointDetector(f)
    cd.DetectCPCUSUM()
    cd.plot()
    plt.title(f'{l} identified changepoints',size=16)
    plt.show()
    f = cd.WriteCPtoXvars()
    fdict[l] = f

# COMMAND ----------

dropped = pd.DataFrame(columns=fdict.keys())

# first to get regressors for regular times
for l, f in fdict.items():
    print(f'reducing {l}')
    f2 = f.deepcopy()
    f2.add_ar_terms(36)
    f.add_seasonal_regressors(
        'month',
        raw=False,
        sincos=True
    )
    f2.add_time_trend()
    f2.diff()

    f2.reduce_Xvars(
        method='pfi',
        estimator='elasticnet',
        cross_validate=True,
        cvkwargs={'k':3},
        dynamic_tuning=fcst_horizon, # optimize on fcst_horizon worth of predictions
        overwrite=False,
    )
    for x in f2.pfi_dropped_vars:
        dropped.loc[x,l] = 1
    for x in f2.reduced_Xvars:
        dropped.loc[x,l] = 0

    plot_reduction_errors(f2)
    plt.title(f'{l} normal times reduction errors',size=12)
    plt.show()

# COMMAND ----------


