# Databricks notebook source
import pandas as pd
import itertools
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 100)

# COMMAND ----------

pd_df = spark.sql('select * from `ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_all_1`')

# COMMAND ----------

display(pd_df)

# COMMAND ----------



# COMMAND ----------

df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()

# COMMAND ----------

df = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()

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
    data,val = df_data.loc[(df_data['ds']>'2015-01-01')&(df_data['ds']<'2022-01-01')],df_data.loc[(df_data['ds']>'2022-01-01')]
    return data,val, df_events

# COMMAND ----------

df.columns = list(map(str.lower,df.columns))
df_events.columns = list(map(str.lower,df_events.columns))

# COMMAND ----------

df['rps'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']
df = df.loc[df['lid']>100]

# COMMAND ----------

data, val, events = prepare_data(df, df_events, origin='ORY', destination='NCE', data_columns=['localdepdt','rps'])

# COMMAND ----------

data = data.groupby(by='ds')['y'].mean().reset_index()
val = val.groupby(by='ds')['y'].mean().reset_index()

# COMMAND ----------

m = Prophet(holidays=events,changepoint_prior_scale=0.001,seasonality_prior_scale=0.01)

# COMMAND ----------

m.fit(data)

# COMMAND ----------

prediction = m.predict(val,['ds'])

# COMMAND ----------

prediction.head()

# COMMAND ----------

prediction[['trend','yhat','holidays','weekly','yearly']].plot()

# COMMAND ----------

fig1 = m.plot(prediction)
fig2 = m.plot_components(prediction)

# COMMAND ----------

val = val.iloc[:90]

# COMMAND ----------

pred = m.predict(val[['ds']])

# COMMAND ----------

pred = pred[['ds','yhat']]

# COMMAND ----------

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# COMMAND ----------

from sklearn.metrics import mean_absolute_error as mae

# COMMAND ----------

mean_absolute_percentage_error(val.y, pred.yhat)

# COMMAND ----------

mae(val.y, pred.yhat)

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

plt.plot(val.y)
plt.plot(pred.yhat)

# COMMAND ----------

df_cv = cross_validation(m, initial='730 days', period='60 days', horizon = '180 days', parallel="processes")
df_p = performance_metrics(df_cv)

# COMMAND ----------

df_p.mdape.mean(), df_p.smape.mean()

# COMMAND ----------

data

# COMMAND ----------

prediction = m.predict(data.loc[data['ds']>'2022-08-01',['ds']])

# COMMAND ----------

df_merged = pd.merge(prediction[['ds','yhat']],
         data.loc[data['ds']>'2022-08-01',['ds','y']],
        on='ds')

# COMMAND ----------

fig1 = m.plot(prediction)
fig2 = m.plot_components(prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC Neural prophet

# COMMAND ----------

data_grouped = data.groupby(by=['ds'])['y'].mean().reset_index()

# COMMAND ----------

data_grouped['ds'] = pd.to_datetime(data_grouped['ds'])

# COMMAND ----------

df_events_nn = df_events[['name','origin','destination','startdate','enddate']]

# COMMAND ----------

df_events_nn["date"] = df_events_nn.apply(
    lambda x: pd.date_range(x["startdate"], x["enddate"]), axis=1
)

# COMMAND ----------

df_events_nn = (
    df_events_nn.explode("date", ignore_index=True)
)

# COMMAND ----------

df_events_nn = df_events_nn.loc[(df_events_nn['origin'].str[:3]=='ORY')|(df_events_nn['destination'].str[3:]=='NCE')]

# COMMAND ----------

df_events_nn = df_events_nn[['name','date']].reset_index(drop=True)

# COMMAND ----------

df_events_nn.columns=['event','ds']

# COMMAND ----------

df_events_nn['ds'] = pd.to_datetime(df_events_nn['ds'])

# COMMAND ----------

m = NeuralProphet(
)

# COMMAND ----------

m = m.add_events(list(df_events_nn.event.unique()),lower_window=-2, upper_window=2)

# COMMAND ----------

history_df = m.create_df_with_events(data_grouped, df_events_nn)

# COMMAND ----------

metrics = m.fit(history_df, freq="30mins")
forecast = m.predict(df=history_df)

# COMMAND ----------

future_df = m.create_df_with_events(data.loc[data['ds']>'2022-08-01',['ds','y']], df_events_nn)

# COMMAND ----------

forecast = m.predict(future_df)

# COMMAND ----------

nn = forecast[['ds','y','yhat1']]

# COMMAND ----------

nn.columns = ['ds','y_nn','yhat_nn']

# COMMAND ----------

df_compare = pd.merge(nn,
        df_merged,
        on='ds')

# COMMAND ----------

df_compare['diff'] = np.abs(df_compare['y']-df_compare['yhat'])
df_compare['diff_nn'] = np.abs(df_compare['y']-df_compare['yhat_nn'])

# COMMAND ----------

df_compare[['ds','y','yhat','yhat_nn','diff','diff_nn']].head(10)

# COMMAND ----------

df_compare[['diff','diff_nn']].sum(axis=0)

# COMMAND ----------

fig_forecast = m.plot(forecast)
fig_components = m.plot_components(forecast)

# COMMAND ----------

fig_model = m.plot_parameters()
