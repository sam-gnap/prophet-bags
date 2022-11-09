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

def prepare_data(df_events, df_rps):
    df_events['ds'] = pd.to_datetime((df_events['StartDate'] + (df_events['EndDate']-df_events['StartDate'])/2).dt.date)
    df_events['lower_window'] = (df_events['StartDate']-df_events['ds']).dt.days
    df_events['upper_window'] = np.abs((df_events['EndDate']-df_events['ds'])).dt.days
    df_events = df_events.loc[(df_events['Origin']=='LGW')|(df_events['Destination']=='AMS'),['Name','ds','lower_window','upper_window']].reset_index(drop=True)
    df_events.columns = ['holiday','ds','lower_window','upper_window']
    lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2021-05-01'},])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    df_events = pd.concat([df_events,lockdowns[['holiday', 'ds', 'lower_window', 'upper_window']]]).reset_index(drop=True)
    df_rps['RPS'] = df_rps['ticketrevenuegbpbudgetnet']/df_rps['lid']
    df_rps = df_rps[['departuredate','RPS']]    
    df_rps.columns = ['ds','y']
    df_rps = df_rps.sort_values(by='ds').reset_index(drop=True)
    rps = df_rps.loc[(df_rps['ds']>'2015-01-01')&(df_rps['ds']<'2022-10-20')]
    return rps, df_events

# COMMAND ----------

df_bags = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`bag_prophet`').toPandas()
df_rps = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`lgwams_rps`').toPandas()
df_rps_all = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`rps_all_sectors`').toPandas()
df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`events`').toPandas()
df_comp = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`competition_lonams`').toPandas()

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

# COMMAND ----------

rps, events = prepare_data(df_events, df_rps)

# COMMAND ----------

events

# COMMAND ----------

comp = df_comp[['dtdeparturedate','n_comp_date','n_comp_morning','n_comp_lunch','n_comp_afternoon','n_comp_evening']].drop_duplicates()
comp.columns = ['ds','n_comp_date','n_comp_morning','n_comp_lunch','n_comp_afternoon','n_comp_evening']

# COMMAND ----------

rps['date'] = pd.to_datetime(rps['ds'].dt.date)
rps = rps.loc[(rps['ds']>'2015-10-01')]

# COMMAND ----------

df = pd.merge(rps,
        comp,
        left_on='date',
        right_on='ds').drop(columns=['date','ds_y'])

# COMMAND ----------

df.columns = ['ds','y', 'n_comp_date', 'n_comp_morning', 'n_comp_lunch',
       'n_comp_afternoon', 'n_comp_evening']

# COMMAND ----------

m = Prophet(holidays=events,changepoint_prior_scale=0.001,seasonality_prior_scale=0.01)
#m.add_regressor('n_comp_date')
#m.add_regressor('n_comp_morning')
#m.add_regressor('n_comp_lunch')
#m.add_regressor('n_comp_afternoon')
#m.add_regressor('n_comp_evening')

# COMMAND ----------

m.fit(rps)

# COMMAND ----------

df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days', parallel="processes")
df_p = performance_metrics(df_cv)

# COMMAND ----------

df_p.mdape.mean(), df_p.smape.mean()

# COMMAND ----------

df_p.mdape.mean(), df_p.smape.mean()

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

df_merged.loc[df_merged['ds']>'2022-07-01'].head(30)

# COMMAND ----------

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale' : [0.01, 0.1, 1.0, 5, 10],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# COMMAND ----------

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(rps)  # Fit model with given params
    df_cv = cross_validation(m, period='180 days', horizon = '365 days', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

# COMMAND ----------

print(tuning_results)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Bagsaa


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


