# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from dateutil.relativedelta import relativedelta
from scalecast.Forecaster import Forecaster
from scalecast import GridGenerator
from scalecast.auxmodels import mlp_stack
from tqdm.notebook import tqdm
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor

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

# COMMAND ----------

rps = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()
df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()

# COMMAND ----------

df = pd.merge(df_bags,
              rps,
              right_on='flightkey',
              left_on='segment')

# COMMAND ----------

df.shape, df_bags.shape, rps.shape

# COMMAND ----------

df['rps'] = df['ticketrevenuegbpbudgetnet']/df['lid_y']
df['yield'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']
df['conversions_23'] = df['23kgbags']/df['seatssold']
df['conversions_15'] = df['15kgbags']/df['seatssold']
df['conversions'] = (df['23kgbags']+df['15kgbags'])/df['seatssold']

# COMMAND ----------

df = df[['sector_x','localdepdt','seatssold','rps','yield','conversions_23','conversions_15','conversions']]

# COMMAND ----------

df.head()

# COMMAND ----------

df = df.loc[df['seatssold']>100]

# COMMAND ----------

df = df.groupby(['sector_x', 'localdepdt']).agg({'seatssold': np.sum, 'rps': np.mean, 
                                            'yield': np.mean, 'conversions_23': np.mean, 'conversions_15': np.mean, 
                                            'conversions': np.mean})

# COMMAND ----------

df = df.reset_index(drop=False)

# COMMAND ----------

orynce_multi = df.loc[df['sector_x']=='ORYNCE']

# COMMAND ----------

# MAGIC %md
# MAGIC ## RPS

# COMMAND ----------

def plot_series(df, x, y):
    f = Forecaster(y=df[x],current_dates=df[y])
    f.plot()
    plt.title('Orig Series',size=16)
    plt.show()

# COMMAND ----------

plot_series(orynce_multi, x='rps', y='localdepdt')

# COMMAND ----------

plot_series(orynce_multi, x='conversions', y='localdepdt')

# COMMAND ----------

plot_series(orynce_multi, x='conversions_15', y='localdepdt')

# COMMAND ----------

plot_series(orynce_multi, x='conversions_23', y='localdepdt')

# COMMAND ----------

orynce_multi_2018 = orynce_multi.loc[orynce_multi['localdepdt']>'2018-01-01']

# COMMAND ----------

f_rps = Forecaster(y=orynce_multi_2018['rps'],current_dates=orynce_multi_2018['localdepdt'])
f_conv = Forecaster(y=orynce_multi_2018['conversions'],current_dates=orynce_multi_2018['localdepdt'])

# COMMAND ----------

figs, axs = plt.subplots(2, 1,figsize=(12,6))
f_rps.plot_acf(ax=axs[0],lags=36)
f_rps.plot_pacf(ax=axs[1],lags=36)
plt.show()

# COMMAND ----------

figs, axs = plt.subplots(2, 1,figsize=(12,6))
f_conv.plot_acf(ax=axs[0],lags=36)
f_conv.plot_pacf(ax=axs[1],lags=36)
plt.show()

# COMMAND ----------

critical_pval = 0.05
print('-'*100)
print('Augmented Dickey-Fuller results:')
stat, pval, _, _, _, _ = f_rps.adf_test(full_res=True)
print('the test-stat value is: {:.2f}'.format(stat))
print('the p-value is {:.4f}'.format(pval))
print('the series is {}'.format('stationary' if pval < critical_pval else 'not stationary'))
print('-'*100)

# COMMAND ----------

critical_pval = 0.05
print('-'*100)
print('Augmented Dickey-Fuller results:')
stat, pval, _, _, _, _ = f_conv.adf_test(full_res=True)
print('the test-stat value is: {:.2f}'.format(stat))
print('the p-value is {:.4f}'.format(pval))
print('the series is {}'.format('stationary' if pval < critical_pval else 'not stationary'))
print('-'*100)

# COMMAND ----------

fcst_length = 60
f_rps.generate_future_dates(fcst_length)
f_rps.set_test_length(.2)
f_rps.add_ar_terms(7)
f_rps.add_AR_terms((4,7))
f_rps.add_seasonal_regressors('month','quarter','week','dayofyear',raw=False,sincos=True)
f_rps.add_seasonal_regressors('dayofweek','is_leap_year','week',raw=False,dummy=True,drop_first=True)
f_rps.add_seasonal_regressors('year')

# COMMAND ----------

fcst_length = 60
f_conv.generate_future_dates(fcst_length)
f_conv.set_test_length(.2)
f_conv.add_ar_terms(7)
f_conv.add_AR_terms((4,7))
f_conv.add_seasonal_regressors('month','quarter','week','dayofyear',raw=False,sincos=True)
f_conv.add_seasonal_regressors('dayofweek','is_leap_year','week',raw=False,dummy=True,drop_first=True)
f_conv.add_seasonal_regressors('year')

# COMMAND ----------



# COMMAND ----------

f_rps.set_estimator('mlr')
f_rps.manual_forecast(dynamic_testing=fcst_length)

# COMMAND ----------

f_conv.set_estimator('mlr')
f_conv.manual_forecast(dynamic_testing=fcst_length)

# COMMAND ----------

def plot_test_export_summaries(f):
    """ exports the relevant statisitcal information and displays a plot of the test-set results for the last model run
    """
    f.plot_test_set(models=f.estimator,ci=True)
    plt.title(f'{f.estimator} test-set results',size=16)
    plt.show()
    return f.export('model_summaries',determine_best_by='TestSetMAPE')[
        [
            'ModelNickname',
            'HyperParams',
            'TestSetMAPE',
            'TestSetR2',
            'InSampleMAPE',
            'InSampleR2'
        ]
    ]



# COMMAND ----------

f_rps.set_estimator('lasso')
lasso_grid = {'alpha':np.linspace(0,2,100)}
f_rps.ingest_grid(lasso_grid)
f_rps.cross_validate(k=3)
f_rps.auto_forecast()

# COMMAND ----------

f_conv.set_estimator('lasso')
lasso_grid = {'alpha':np.linspace(0,2,100)}
f_conv.ingest_grid(lasso_grid)
f_conv.cross_validate(k=3)
f_conv.auto_forecast()

# COMMAND ----------

f_rps.set_estimator('ridge')
f_rps.ingest_grid(lasso_grid)
f_rps.cross_validate(k=3)
f_rps.auto_forecast()

# COMMAND ----------

f_conv.set_estimator('ridge')
f_conv.ingest_grid(lasso_grid)
f_conv.cross_validate(k=3)
f_conv.auto_forecast()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

GridGenerator.get_example_grids()

# COMMAND ----------

f_rps.add_sklearn_estimator(BaggingRegressor,'bagging')
f_rps.add_sklearn_estimator(StackingRegressor,'stacking')

# COMMAND ----------

f_conv.add_sklearn_estimator(BaggingRegressor,'bagging')
f_conv.add_sklearn_estimator(StackingRegressor,'stacking')

# COMMAND ----------

f_rps.set_estimator('elasticnet')
f_rps.ingest_grid(lasso_grid)
f_rps.cross_validate(k=3)
f_rps.auto_forecast()

# COMMAND ----------

f_conv.set_estimator('elasticnet')
f_conv.ingest_grid(lasso_grid)
f_conv.cross_validate(k=3)
f_conv.auto_forecast()

# COMMAND ----------

plot_test_export_summaries(f_rps)

# COMMAND ----------

plot_test_export_summaries(f_conv)

# COMMAND ----------

f_rps.set_estimator('lightgbm')
lightgbm_grid = {
    'n_estimators':[150,200,250],
    'boosting_type':['gbdt','dart','goss'],
    'max_depth':[1,2,3],
    'learning_rate':[0.001,0.01,0.1],
    'reg_alpha':np.linspace(0,1,5),
    'reg_lambda':np.linspace(0,1,5),
    'num_leaves':np.arange(5,50,5),
}
f_rps.ingest_grid(lightgbm_grid)
f_rps.limit_grid_size(100,random_seed=2)
f_rps.cross_validate(k=3)
f_rps.auto_forecast()

# COMMAND ----------

f_rps.set_estimator('stacking')
results = f_rps.export('model_summaries')
estimators = [
    'lightgbm',
    'mlr'
]

mlp_stack(f_rps,estimators,call_me='stacking')

# COMMAND ----------

f.plot_test_set(
    models=['mlr'],
    order_by='TestSetRMSE',
    include_train=False,
)
plt.title('All Models Test Performance - test set obs only',size=16)
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


