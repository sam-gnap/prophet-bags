# Databricks notebook source
import pandas as pd
import itertools
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
pd.set_option('display.max_rows', 100)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import current_date
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC The best result so far has changepoint_prior_scale of 0.001 seasonality_prior_scale of 0.01 with seasonality_mode being multiplicative 15.456195 RMSE.

# COMMAND ----------

def morning_lunch_afternoon_evening(hour):
    if hour> 0 and hour<9:
        return 'morning'
    elif hour>=9 and hour<14:
        return 'lunch'
    elif hour>=14 and hour <20:
        return 'afternoon'
    else:
        return 'evening'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load holidays 

# COMMAND ----------

df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()
df_events['ds'] = pd.to_datetime((df_events['StartDate'] + (df_events['EndDate']-df_events['StartDate'])/2).dt.date)
df_events['lower_window'] = (df_events['StartDate']-df_events['ds']).dt.days
df_events['upper_window'] = np.abs((df_events['EndDate']-df_events['ds'])).dt.days
df_events = df_events[['Name','ds','lower_window','upper_window','Origin','Destination']]
df_events.columns = ['holiday','ds','lower_window','upper_window','Origin','Destination']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Add lockdown effects 
# MAGIC We should make this more elaborate where we add exact measures by country! For example, we still do have restrictions in Netherlands during summer 2021!!!

# COMMAND ----------



# COMMAND ----------

lockdowns = pd.DataFrame([
{'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2021-05-01','Origin':None,'Destination':None},])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

# COMMAND ----------

events = pd.concat([df_events,lockdowns[['holiday', 'ds', 'lower_window', 'upper_window', 'Origin', 'Destination']]]).reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RPS for sectors

# COMMAND ----------

df = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet")

# COMMAND ----------

df = df.filter(df.lid > 100)
df = df.withColumn('rps', (F.col("ticketrevenuegbpbudgetnet") / F.col("lid")))

# COMMAND ----------

condition_sector = df.groupBy("sector")\
                  .agg(F.count("*").alias("cnt"))\
                  .filter("cnt > 1000").select("sector")

# COMMAND ----------

results_df = df.join(condition_sector, on=("sector"))

# COMMAND ----------

df_r = results_df.toPandas()

# COMMAND ----------

df_r.shape

# COMMAND ----------

df_r.head()

# COMMAND ----------

df_r.sector.nunique()

# COMMAND ----------

df_r.localdepdt.max()

# COMMAND ----------

# MAGIC %md
# MAGIC ### RPS prediction at scale

# COMMAND ----------

to_predict = results_df.select('sector','departuredate','rps')
to_predict = to_predict.toDF(*['sector','ds','y'])
to_predict = to_predict.na.drop()

# COMMAND ----------

def forecast_sectors(history_pd: pd.DataFrame) -> pd.DataFrame: 
    # instantiate the model, configure the parameters
    history = history_pd.loc[history_pd['ds']<'2022-08-01']
    to_predict = history_pd.loc[history_pd['ds']>'2022-08-01',['ds']]
    if not to_predict.empty:
        model = Prophet(holidays=events,
                changepoint_prior_scale=0.001,
                seasonality_prior_scale=0.01,
                seasonality_mode='multiplicative')
        sector = history['sector'].iloc[0]
        events.loc[(events['Origin']==sector[:3])|(events['Destination']==sector[3:])\
                                  |(events['Origin'].isnull())&(events['Destination'].isnull())
                                  ,['holiday','ds','lower_window','upper_window']].reset_index(drop=True)
        
        # fit the model
        model.fit(history)

        # make predictions
        forecast_pd = model.predict(to_predict)

        f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')

        # get relevant fields from history
        h_pd = history_pd[['ds','sector','y']].set_index('ds')

        # join history and forecast
        results_pd = f_pd.join( h_pd, how='left' )
        results_pd.reset_index(level=0, inplace=True)

        # get store & item from incoming data set
        results_pd['sector'] = sector
        # --------------------------------------

        # return expected dataset
        return results_pd[ ['ds', 'sector', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ] 
    else:
        d = {'ds': [history['ds'].iloc[0]], 'sector': [history['sector'].iloc[0]], 'y': -1, 'yhat': -1, 'yhat_upper': -1, 'yhat_lower': -1}        
        results_pd =pd.DataFrame(data=d)
        return results_pd

# COMMAND ----------

 result_schema =StructType([
  StructField('ds',TimestampType()),
  StructField('sector',StringType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

results = (
    to_predict
    .groupBy('sector')
    .applyInPandas(forecast_sectors, schema=result_schema)
    .withColumn('training_date', current_date())
    )
results = results.na.drop()
results.createOrReplaceTempView('new_forecasts')

# COMMAND ----------

results.show()

# COMMAND ----------

df = results.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

results.write.mode("append").saveAsTable("ds_data_analytics.data_analytics_sandbox.bag_prophet_forecast")

# COMMAND ----------

df = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`bag_prophet_forecast`').toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

df.sector.nunique()

# COMMAND ----------


