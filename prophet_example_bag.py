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

def prepare_data(df_data, df_events, origin, destination, data_columns=['departuredate','RPS']):
    df_data = df_data.loc[(df_data['sector'].str[:3]==origin)&(df_data['sector'].str[3:]==destination)]
    df_events['ds'] = pd.to_datetime((df_events['startdate'] + (df_events['enddate']-df_events['startdate'])/2).dt.date)
    df_events['lower_window'] = (df_events['startdate']-df_events['ds']).dt.days
    df_events['upper_window'] = np.abs((df_events['enddate']-df_events['ds'])).dt.days
    df_events = df_events.loc[(df_events['origin']==origin)|(df_events['destination']==destination),['name','ds','lower_window','upper_window']].reset_index(drop=True)
    df_events.columns = ['holiday','ds','lower_window','upper_window']
    lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2021-05-01'},])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    df_events = pd.concat([df_events,lockdowns[['holiday', 'ds', 'lower_window', 'upper_window']]]).reset_index(drop=True)
    df_data = df_data[data_columns]
    df_data.columns = ['ds','y']
    df_data = df_data.sort_values(by='ds').reset_index(drop=True)
    data = df_data.loc[(df_data['ds']>'2015-01-01')&(df_data['ds']<'2022-10-20')]
    return data, df_events

# COMMAND ----------

df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()

# COMMAND ----------

df_bags = df_bags.loc[df_bags['sold']>20]
df_bags['bags'] = df_bags['23kgbags']+df_bags['15kgbags']
df_bags['bags_conversion'] = df_bags['bags']/df_bags['sold']
df_bags.loc[df_bags['bags_conversion']>1, 'bags_conversion'] = 1

# COMMAND ----------

#df_bags = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_example`').toPandas()
df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()
df_countries = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`example_r`').toPandas()

# COMMAND ----------

df_bags.columns = list(map(str.lower,df_bags.columns))
df_events.columns = list(map(str.lower,df_events.columns))
df_countries.columns = list(map(str.lower,df_countries.columns))

# COMMAND ----------

df_countries = df_countries[['sector','region2','countrypair']]

# COMMAND ----------

df_bags.segment.nunique()

# COMMAND ----------

df_bags = pd.merge(df_countries,
        df_bags,
        on='sector')

# COMMAND ----------

df_bags.segment.nunique()

# COMMAND ----------

def prepare_data_country(df_data, df_events,data_columns=['departuredate','RPS']):
    df_events['ds'] = pd.to_datetime((df_events['startdate'] + (df_events['enddate']-df_events['startdate'])/2).dt.date)
    df_events['lower_window'] = (df_events['startdate']-df_events['ds']).dt.days
    df_events['upper_window'] = np.abs((df_events['enddate']-df_events['ds'])).dt.days
    df_events = df_events.loc[(df_events['origin']==origin)|(df_events['destination']==destination),['name','ds','lower_window','upper_window']].reset_index(drop=True)
    df_events.columns = ['holiday','ds','lower_window','upper_window']
    lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2021-05-01'},])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    df_events = pd.concat([df_events,lockdowns[['holiday', 'ds', 'lower_window', 'upper_window']]]).reset_index(drop=True)
    df_data = df_data[data_columns]
    df_data.columns = ['ds','y']
    df_data = df_data.sort_values(by='ds').reset_index(drop=True)
    data = df_data.loc[(df_data['ds']>'2015-01-01')&(df_data['ds']<'2022-10-20')]
    return data, df_events

# COMMAND ----------

df_bags = df_bags[['std','bags_conversion']].groupby('std').mean()

# COMMAND ----------

df_bags = df_bags.reset_index()

# COMMAND ----------

df_bags.columns = ['ds','y']

# COMMAND ----------

data = df_bags.loc[(df_bags['ds']>'2015-01-01')&(df_bags['ds']<'2022-10-20')]

# COMMAND ----------

m = Prophet(changepoint_prior_scale=0.001,seasonality_prior_scale=0.01)

# COMMAND ----------

m.fit(data)

# COMMAND ----------

df_cv = cross_validation(m, initial='730 days', period='60 days', horizon = '180 days', parallel="processes")
df_p = performance_metrics(df_cv)

# COMMAND ----------

df_p.mdape.mean(), df_p.smape.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC UK: (0.2608908743146285, 0.3177562741194462)
# MAGIC ALL: (0.22424181067286167, 0.2805076447144276)

# COMMAND ----------

df_p.mdape.mean(), df_p.smape.mean()

# COMMAND ----------

prediction = m.predict(df.loc[df['ds']>'2022-08-01',['ds']])

# COMMAND ----------

df_merged = pd.merge(prediction[['ds','yhat']],
         df.loc[df['ds']>'2022-08-01',['ds','y']],
        on='ds')

# COMMAND ----------

fig1 = m.plot(prediction)
fig2 = m.plot_components(prediction)

# COMMAND ----------

df_merged.head(50)

# COMMAND ----------


