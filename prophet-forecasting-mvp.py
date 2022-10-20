# Databricks notebook source
import pandas as pd
from prophet import Prophet
import numpy as np

# COMMAND ----------

df_spark = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`bag_prophet`')
df_bags = df_spark.toPandas()

# COMMAND ----------

df_spark = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`lgwams_rps`')
df_rps = df_spark.toPandas()

# COMMAND ----------

df_spark = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`events`')
df_events = df_spark.toPandas()

# COMMAND ----------

df_events['ds'] = pd.to_datetime((df_events['StartDate'] + (df_events['EndDate']-df_events['StartDate'])/2).dt.date)

# COMMAND ----------

df_events['lower_window'] = (df_events['StartDate']-df_events['ds']).dt.days

# COMMAND ----------

df_events['upper_window'] = np.abs((df_events['EndDate']-df_events['ds'])).dt.days

# COMMAND ----------

df_events.head()

# COMMAND ----------

df_events = df_events.loc[(df_events['Origin']=='LGW')|(df_events['Destination']=='AMS'),['Name','ds','lower_window','upper_window']].reset_index(drop=True)

# COMMAND ----------

df_events.columns = ['holiday','ds','lower_window','upper_window']

# COMMAND ----------

lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2021-05-01'},
    {'holiday': 'lockdown_2', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
lockdowns

# COMMAND ----------

df_events = pd.concat([df_events,lockdowns[['holiday','ds','lower_window','upper_window']]])

# COMMAND ----------

df_rps['RPS'] = df_rps['ticketrevenuegbpbudgetnet']/df_rps['lid']

# COMMAND ----------

df_rps = df_rps[['departuredate','RPS']]

# COMMAND ----------

df_rps.columns = ['ds','y']

# COMMAND ----------

m = Prophet(holidays=df_events)

# COMMAND ----------

forecast = m.fit(df_rps)

# COMMAND ----------

future = m.make_future_dataframe(periods=365)

# COMMAND ----------

forecast_future = forecast.predict(future)

# COMMAND ----------

fig = m.plot_components(forecast_future)

# COMMAND ----------

fig = m.plot_components(forecast_future)

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot

# COMMAND ----------

fig = m.plot(forecast_future)
a = add_changepoints_to_plot(fig.gca(), m, forecast_future)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df['sector'] = df.Segment.str[8:14]

# COMMAND ----------

df.head()

# COMMAND ----------

df['seats_sold'] = df[['Seatsold_45','Seatsold_120','Seatsold_osd']].sum(axis=1)
df['bags'] = df[['Total_Bags_45','Total_Bags_120','Total_Bags_osd']].sum(axis=1)

# COMMAND ----------

df.head()

# COMMAND ----------

df['conversion'] = df['bags']/df['seats_sold']

# COMMAND ----------

def make_forecast(df, period, sector, bag):
    dfs = df.loc[df['BagFilter']==bag,['sector','FltDate','conversion']]
    dfs.columns = ['sector','ds','conversion']
    dfs = dfs.sort_values(by=['sector','ds']).reset_index(drop=True)
    dfs = pd.pivot_table(dfs,
              index=['sector','ds'],
              values='conversion').reset_index()
    dfs = dfs.loc[dfs['sector']==sector,['ds','conversion']]
    dfs.columns = ['ds','y']
    m = Prophet()
    m.fit(dfs)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)

# COMMAND ----------

make_forecast(df, 165, 'AMSLGW', '15kg')

# COMMAND ----------

make_forecast(df, 165, 'AMSLGW', '23kg')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


