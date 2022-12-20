# Databricks notebook source
import pandas as pd
import numpy as np
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

df['rps'] = df['ticketrevenuegbpbudgetnet']/df['lid_y']
df['yield'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']
df['conversions_23'] = df['23kgbags']/df['seatssold']
df['conversions_15'] = df['15kgbags']/df['seatssold']
df['conversions'] = (df['23kgbags']+df['15kgbags'])/df['seatssold']

# COMMAND ----------

df_bags = pre_process_bags(df_bags)
orynce = df_bags.loc[df_bags['sector']=='ORYNCE']
df = df[['sector_x','localdepdt','seatssold','rps','yield','conversions_23','conversions_15','conversions']]
df = df.loc[df['seatssold']>100]
df = df.groupby(['sector_x', 'localdepdt']).agg({'seatssold': np.sum, 'rps': np.mean, 
                                            'yield': np.mean, 'conversions_23': np.mean, 'conversions_15': np.mean, 
                                            'conversions': np.mean})
df = df.reset_index(drop=False)

# COMMAND ----------

orynce = df.loc[df['sector_x']=='ORYNCE',['localdepdt','rps','conversions']].reset_index(drop=True)

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

figs, axs = plt.subplots(2, 1,figsize=(9,9))
f.plot_acf(ax=axs[0],title='ACF',lags=26,color='black')
f.plot_pacf(ax=axs[1],title='PACF',lags=26,color='#B2C248')
plt.show()

# COMMAND ----------

f.set_test_length(.15) # all models will be tested out of sample
f.add_time_trend()
f.add_seasonal_regressors('weekday',raw=False,sincos=True)
f.add_seasonal_regressors('month',raw=False,sincos=True)
f.add_seasonal_regressors('week',raw=False,sincos=True)
f.add_seasonal_regressors('weekday',raw=False,sincos=True)
f.add_ar_terms(365)

f.set_estimator('lasso')
f.manual_forecast(alpha=0.2,dynamic_testing=14)

f.set_estimator('ridge')
f.manual_forecast(alpha=0.2,dynamic_testing=14)

f.set_estimator('elasticnet')
f.manual_forecast(alpha=0.2,l1_ratio=0.5,dynamic_testing=14)

f.set_estimator('sgd')
f.proba_forecast(alpha=0.2,l1_ratio=0.5,dynamic_testing=14)

# COMMAND ----------

results = f.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------

f.plot()
plt.show()

# COMMAND ----------

from scalecast.SeriesTransformer import SeriesTransformer

# COMMAND ----------

f_trans = Forecaster(
    y = orynce.rps,
    current_dates = orynce.localdepdt,
    future_dates = 90,
    freq='d'
)

# COMMAND ----------

f_trans.set_test_length(.15) # all models will be tested out of sample
f_trans.add_time_trend()
f_trans.add_seasonal_regressors('weekday',raw=False,sincos=True)
f_trans.add_seasonal_regressors('month',raw=False,sincos=True)
f_trans.add_seasonal_regressors('week',raw=False,sincos=True)
f_trans.add_seasonal_regressors('weekday',raw=False,sincos=True)
f_trans.add_ar_terms(7)

# COMMAND ----------

transformer = SeriesTransformer(f_trans)

# COMMAND ----------

# these will all be reverted later after forecasts have been called
f_trans = transformer.DiffTransform(1)
f_trans = transformer.DiffTransform(52)
f_trans = transformer.DetrendTransform()

# COMMAND ----------

f_trans.plot()
plt.show()

# COMMAND ----------

f_trans.set_estimator('lasso')
f_trans.manual_forecast(alpha=0.2,dynamic_testing=90)

f_trans.set_estimator('ridge')
f_trans.manual_forecast(alpha=0.2,dynamic_testing=90)

# COMMAND ----------

results = f_trans.export(cis=True)
results.keys()
results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------

f_trans.plot()
plt.show()

# COMMAND ----------

f_trans

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-linear models

# COMMAND ----------

f.set_estimator('rf')
f.proba_forecast(max_depth=2,dynamic_testing=90)

f.set_estimator('gbt')
f.proba_forecast(max_depth=2,dynamic_testing=90)

f.set_estimator('xgboost')
f.manual_forecast(gamma=1,dynamic_testing=90)

f.set_estimator('lightgbm')
f.manual_forecast(max_depth=2,dynamic_testing=90)

f.set_estimator('knn')
f.manual_forecast(n_neighbors=5,dynamic_testing=90)

f.set_estimator('mlp')
f.proba_forecast(hidden_layer_sizes=(50,50),solver='lbfgs',dynamic_testing=90)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

### Stacking

# COMMAND ----------



# COMMAND ----------

f.add_sklearn_estimator(StackingRegressor,'stacking')

# COMMAND ----------

estimators = [
    ('elasticnet',ElasticNet(alpha=0.2)),
    ('xgboost',XGBRegressor(gamma=1)),
    ('gbt',GradientBoostingRegressor(max_depth=2)),
]

final_estimator = LGBMRegressor()

f.set_estimator('stacking')
f.manual_forecast(
    estimators=estimators,
    final_estimator=final_estimator,
    dynamic_testing=90
)

# COMMAND ----------

from scalecast.auxmodels import mlp_stack

# COMMAND ----------

mlp_stack(f,model_nicknames=['elasticnet','lightgbm','xgboost','knn'],dynamic_testing=13)

# COMMAND ----------

f.plot_test_set(models=['stacking','mlp_stack'],ci=True,order_by='LevelTestSetMAPE')
plt.show()

# COMMAND ----------

f.plot(models=['stacking','mlp_stack'],ci=True,order_by='LevelTestSetMAPE')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prophet

# COMMAND ----------

f.set_estimator('prophet')
f.manual_forecast()

# COMMAND ----------

f.plot_test_set(models='prophet',ci=True)
plt.show()

# COMMAND ----------

f.plot(models='prophet',ci=True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multivariate Forecasting

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

results = f.export(cis=True)
results.keys()

# COMMAND ----------

for k, df in results.items():
    print(f'{k} has these columns : {df.columns}')

# COMMAND ----------

results['model_summaries'][['ModelNickname','HyperParams','TestSetMAPE','InSampleMAPE']]

# COMMAND ----------


