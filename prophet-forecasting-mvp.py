# Databricks notebook source
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.utilities import regressor_coefficients 
pd.set_option('display.max_rows', 100)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )

# COMMAND ----------

# MAGIC %md
# MAGIC The best result so far has changepoint_prior_scale of 0.001 seasonality_prior_scale of 0.01 with seasonality_mode being multiplicative 15.456195 RMSE.

# COMMAND ----------

df_comp = df_comp.sort_values(by='dtdeparturedate').reset_index(drop=True)
df_comp['hour'] = df_comp['tmdeparturetime'].dt.hour
df_comp['n_comp_date'] = df_comp.groupby(by=['dtdeparturedate'])['vchflightnumber'].transform('count')
df_comp['n_comp_hour'] = df_comp.groupby(by=['dtdeparturedate','hour'])['vchflightnumber'].transform('count')
df_comp['hour'] = df_comp.hour.astype('int')
df_comp['tod'] = df_comp['hour'].apply(morning_lunch_afternoon_evening)
df_comp = pd.get_dummies(df_comp, columns=['tod'], prefix='', prefix_sep='')
df_comp['n_comp_morning'] = df_comp.groupby(by=['dtdeparturedate'])['morning'].transform('sum')
df_comp['n_comp_lunch'] = df_comp.groupby(by=['dtdeparturedate'])['lunch'].transform('sum')
df_comp['n_comp_afternoon'] = df_comp.groupby(by=['dtdeparturedate'])['afternoon'].transform('sum')
df_comp['n_comp_evening'] = df_comp.groupby(by=['dtdeparturedate'])['evening'].transform('sum')
comp = df_comp[['dtdeparturedate','n_comp_date','n_comp_morning','n_comp_lunch','n_comp_afternoon','n_comp_evening']].drop_duplicates()
comp.columns = ['ds','n_comp_date','n_comp_morning','n_comp_lunch','n_comp_afternoon','n_comp_evening']

# COMMAND ----------

def get_mape(df):
    return np.mean(np.abs((df['y']-df['yhat'])/df['y']))

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

def get_holidays(events, sector):
    events = events.loc[(events['Origin']==sector[:3])|(events['Destination']==sector[3:]),
           ['Name','StartDate','EndDate','Origin','Destination','HalfOfDaysBetween']].sort_values(by=['Name','StartDate']).reset_index(drop=True)
    lockdowns = pd.DataFrame([{'Name': 'lockdown', 
                               'StartDate': '2020-03-21', 
                               'EndDate': '2021-05-01', 
                               'Origin':sector[:3],
                               'Destination':sector[3:],
                               'HalfOfDaysBetween':0},])
    events = pd.concat([events,lockdowns]).reset_index(drop=True)
    events['StartDate'] = pd.to_datetime(events['StartDate'])
    events['EndDate'] = pd.to_datetime(events['EndDate'])
    events = events.drop_duplicates()
    d = events['EndDate'].sub(events['StartDate']).dt.days
    df1 = events.reindex(events.index.repeat(d))
    i = df1.groupby(level=0).cumcount() + 1
    df1['date'] = df1['StartDate'] + pd.to_timedelta(i, unit='d')
    df1 = df1[['Name','date','HalfOfDaysBetween']].drop_duplicates()
    df1['value'] = 1
    df1['HalfOfDaysBetween_plus1'] = round(df1['HalfOfDaysBetween'])+1
    df1['lower_window'] = -df1['HalfOfDaysBetween_plus1']
    df1['upper_window'] = df1['HalfOfDaysBetween_plus1']
    df1 = df1[['Name','date','lower_window','upper_window']]
    df1.columns = ['holiday','ds','lower_window','upper_window']
    return df1

# COMMAND ----------

df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()
df_rps = spark.read.parquet(
    "dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet"
).toPandas()

# COMMAND ----------

df_rps = df_rps.loc[(df_rps['lid']>100)&(df_rps['seatssold']>50)]
df_rps["rps"] = df_rps["ticketrevenuegbpbudgetnet"] / df_rps["lid"]
df_rps["yield"] = df_rps["ticketrevenuegbpbudgetnet"] / df_rps["seatssold"]

# COMMAND ----------

sector = 'ORYNCE'

# COMMAND ----------

start_date = pd.to_datetime('2022-08-15')
for i in range(3):
    end_date = start_date + pd.Timedelta("14 day")
    print(start_date, end_date)
    holidays = get_holidays(df_events, sector)
    rps = df_rps.loc[df_rps['sector']==sector,['departuredate','rps']]
    rps.columns = ['ds','y']
    rps = rps.sort_values(by=['ds'])
    rps_test = rps.loc[(rps['ds']>=start_date)&(rps['ds']<=end_date)]
    rps = rps.loc[(rps['ds']>='2017-01-01')&(rps['ds']<start_date)]
    m = Prophet(holidays=holidays,
                changepoint_prior_scale=0.001,
                seasonality_prior_scale=0.01,
               seasonality_mode='multiplicative')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=12)
    m.add_seasonality(name='quarterly', period=30.5, fourier_order=4)
    m.add_seasonality(name='hourly', period=30.5, fourier_order=24)
    m.fit(rps)
    predictions = m.predict(rps_test[['ds']])
    predictions_merged = pd.merge(rps_test,
                             predictions,
                             on=['ds'])
    print("*****************************")
    print(get_mape(predictions_merged))
    print("*****************************")
    start_date = end_date

# COMMAND ----------

predictions = m.predict(rps_test[['ds']])

# COMMAND ----------

from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

# COMMAND ----------

rps['ds_hour'] = rps['ds'].dt.round(freq='CBH')
rps_h = rps.groupby(by=['ds_hour'])['y'].mean().reset_index()

# COMMAND ----------

ts = pd.Timestamp(2022, 8, 5, 16)

# COMMAND ----------

ts

# COMMAND ----------

ts + pd.offsets.CustomBusinessHour(start='06:00', end='13:00')

# COMMAND ----------

# specify dataset information
metadata = MetadataParam(
 time_col="ds_hour",  # name of the time column ("date" in example above)
 value_col="y",  # name of the value column ("sessions" in example above)
 freq="BH"
)

# COMMAND ----------

forecaster = Forecaster()  # Creates forecasts and stores the result
result = forecaster.run_forecast_config(  # result is also stored as `forecaster.forecast_result`.
 df=rps_h,
 config=ForecastConfig(
     model_template=ModelTemplateEnum.AUTO.name,
     forecast_horizon=14,  # forecasts 365 steps ahead
     coverage=0.95,         # 95% prediction intervals
     metadata_param=metadata
 )
)

# COMMAND ----------

import plotly

# COMMAND ----------

result

# COMMAND ----------

ts = result.timeseries
fig = ts.plot()
plotly.io.show(fig)

# COMMAND ----------

ts

# COMMAND ----------

print(ts.value_stats)

# COMMAND ----------

ts.df.tail(100)

# COMMAND ----------

rps

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

m = Prophet(holidays=holidays,changepoint_prior_scale=0.001,seasonality_prior_scale=0.01)
#m.add_regressor('n_comp_date')
#m.add_regressor('n_comp_morning')
#m.add_regressor('n_comp_lunch')
#m.add_regressor('n_comp_afternoon')
#m.add_regressor('n_comp_evening')
m.fit(rps)

# COMMAND ----------

predictions = m.predict(rps_test[['ds']])[['ds','yhat','yhat_lower','yhat_upper']]

# COMMAND ----------

predictions_merged = pd.merge(rps_test,
                             predictions,
                             on=['ds'])

# COMMAND ----------

plt.plot(predictions_merged['ds'], predictions_merged[['y','yhat']])
plt.show()

# COMMAND ----------

get_mape(predictions_merged)

# COMMAND ----------

fig1 = m.plot(predictions_merged)

# COMMAND ----------



# COMMAND ----------

regressor_coefficients(m)

# COMMAND ----------

predictions_merged.loc[predictions_merged['ds']>'2022-09-01'].head(40)

# COMMAND ----------

ts.df.tail(20)

# COMMAND ----------

# DBTITLE 1,Results 
Without any additional regressors we get: 
    MDAPE: 0.19226
    SMAPE: 0.25942
Using all additional regressors we get: 
    MDAPE: 0.18308
    SMAPE: 0.25253

# COMMAND ----------

prediction = m.predict(rps.loc[rps['ds']>'2022-06-01',['ds']])

# COMMAND ----------

df_merged = pd.merge(prediction[['ds','yhat']],
         rps.loc[rps['ds']>'2022-06-01',['ds','y']],
        on='ds')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_cv = cross_validation(m, initial='1460 days', period='65 days', horizon = '14 days', parallel="processes")
df_p = performance_metrics(df_cv)
df_p = performance_metrics(df_cv)
df_p.tail()

