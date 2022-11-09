# Databricks notebook source
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts import utils
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
)
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
# for reproducibility
np.random.seed(1)

# COMMAND ----------

df = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()

# COMMAND ----------

df = df.loc[df['lid']>100]
df = df.loc[df['localdepdt']>'2015-10-01']
df['rps'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']

# COMMAND ----------

lgwams = df.loc[df['sector']=='ORYNCE',['localdepdt','rps']]
amslgw = df.loc[df['sector']=='NCEORY',['localdepdt','rps']]
lgwams = lgwams.groupby('localdepdt')['rps'].mean().reset_index()
amslgw = amslgw.groupby('localdepdt')['rps'].mean().reset_index()
lgwams = TimeSeries.from_dataframe(lgwams, time_col='localdepdt',fill_missing_dates=False, freq='d')
amslgw = TimeSeries.from_dataframe(amslgw, time_col='localdepdt',fill_missing_dates=False, freq='d')

# COMMAND ----------

lgwams_sub = utils.missing_values.extract_subseries(lgwams)
amslgw_sub = utils.missing_values.extract_subseries(amslgw)

# COMMAND ----------

lgwams_filled = utils.missing_values.fill_missing_values(lgwams)
lgwams_filled_scaler = Scaler()
lgwams_filled_scaled = lgwams_filled_scaler.fit_transform(lgwams_filled)
train_lgwams, val_lgwams = lgwams_filled_scaled[:-70], lgwams_filled_scaled[-70:]

# COMMAND ----------

amslgw_filled = utils.missing_values.fill_missing_values(amslgw)
amslgw_filled_scaler = Scaler()
amslgw_filled_scaled = amslgw_filled_scaler.fit_transform(amslgw_filled)
train_amslgw, val_amslgw = amslgw_filled_scaled[:-70], amslgw_filled_scaled[-70:]

# COMMAND ----------

# MAGIC %md
# MAGIC ### One series

# COMMAND ----------

encoders = {"datetime_attribute": {"past": ["dayofweek","month", "year"]}, "cyclic": {"past": ["dayofweek","month","week","day"]}, "transformer": Scaler()}

# COMMAND ----------

model_lgwams = NBEATSModel(
    input_chunk_length=2000, output_chunk_length=70, n_epochs=300, random_state=0, add_encoders=encoders, pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": 1
    },
)
model_lgwams.fit(train_amslgw, verbose=True)

# COMMAND ----------

pred.pd_dataframe()

# COMMAND ----------

pred = model_lgwams.predict(series=train_amslgw, n=70)
#pred = model_lgwams.predict(n=70)

val_amslgw.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(val_amslgw, pred)))

# COMMAND ----------

# scale back:
pred_rescaled = amslgw_filled_scaler.inverse_transform(pred)

plt.figure(figsize=(10, 6))
amslgw_filled[-70:].plot(label="actual (air)")
pred_rescaled.plot(label="forecast (air)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multiple series

# COMMAND ----------

encoders = {"datetime_attribute": {"past": ["dayofweek","month", "year"]}, "cyclic": {"past": ["dayofweek","month","week","day"]}, "transformer": Scaler()}

# COMMAND ----------

model_lgwams = NBEATSModel(
    input_chunk_length=2000, output_chunk_length=70, n_epochs=300, random_state=0, add_encoders=encoders, pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": 1
    },
)
model_lgwams.fit([train_amslgw,train_lgwams], verbose=True)

# COMMAND ----------

pred = model_lgwams.predict(series=train_lgwams, n=70)
#pred = model_lgwams.predict(n=70)

val_lgwams[-70:].plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(val_lgwams, pred)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding in holidays

# COMMAND ----------

df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()

# COMMAND ----------

orynce_events = df_events.loc[(df_events['Origin']=='NCE')|(df_events['Destination']=='ORY'),['Name','StartDate','EndDate']]
lockdowns = pd.DataFrame([
{'Name': 'lockdown', 'StartDate': '2020-03-21', 'EndDate': '2021-05-01'},])
orynce_events = pd.concat([orynce_events,lockdowns]).reset_index(drop=True)
orynce_events['StartDate'] = pd.to_datetime(orynce_events['StartDate'])
orynce_events['EndDate'] = pd.to_datetime(orynce_events['EndDate'])
d = orynce_events['EndDate'].sub(orynce_events['StartDate']).dt.days
df1 = orynce_events.reindex(orynce_events.index.repeat(d))
i = df1.groupby(level=0).cumcount() + 1

df1['date'] = df1['StartDate'] + pd.to_timedelta(i, unit='d')
df1 = df1[['Name','date']].drop_duplicates()
df1['value'] = 1
df1 = pd.pivot_table(df1,
              index='date',
              columns='Name',
              values='value').reset_index()
df1 = df1.replace(np.nan,0)
idx = pd.date_range('2015-10-01', '2023-10-20')
df1.index = pd.DatetimeIndex(df1.date)
df1 = df1.reindex(idx, fill_value=0)
df1 = df1.reset_index()
df1 = df1.drop(columns=['date'])

# COMMAND ----------

df1.columns = ['date', 'Air France strike', 'Autumn School Holiday (FRZoneB)',
       'Batimat Exhibition (Paris)', 'Christmas School Holiday (FRZoneB)',
       'Cop', 'Diabetes UK Professional Conference', 'Easter', 'Equip Auto',
       'EquipHotel Paris', 'Eurosatory', 'Foire De Paris (Hors-Serie Maison)',
       'IFTM Top Resa', 'ITM Conference', 'Maison&Objet Paris I',
       'Maison&Objet Paris II', "Mondial de l'Automobile Paris",
       'MotoGP - France', 'Paris Air Show',
       "Paris Haute Couture's Fashion Week (Spring/Summer)",
       "Paris Men's Fashion Week (Autumn/Winter)",
       "Paris Men's Fashion Week - RTW (Spring/Summer)", 'Paris Photo',
       "Paris Women's Fashion Week - RTW (Spring/Summer)",
       "Paris Women's Fashion Week - Ready to Wear (Autumn/Winter)",
       'Premiere Vision (February)', 'Premiere Vision (September)',
       'Roland Garros (The French Open)', 'Ryder Cup', 'SIAL',
       "Salon International De L'agriculture",
       "Salon International de l’Aéronautique et de l’Espa",
       'Salon Nautique de Paris', 'Salon du Chocolat', 'Salon du Mariage',
       'Solidays', 'Spring School Holiday (FRZoneB)',
       'Spring School Holiday Mid (FRZoneB)',
       'Summer School Holiday End (France)', 'Winter School Holiday (FRZoneB)',
       'lockdown']

# COMMAND ----------

events = TimeSeries.from_dataframe(df1, time_col='date',fill_missing_dates=False, freq='d')

# COMMAND ----------

# MAGIC %md
# MAGIC ### One series

# COMMAND ----------

encoders = {"datetime_attribute": {"past": ["dayofweek","month", "year"]}, "cyclic": {"past": ["dayofweek","month","week","day"]}, "transformer": Scaler()}

# COMMAND ----------

import gc

# COMMAND ----------

gc.collect()

# COMMAND ----------

model_lgwams = NBEATSModel(
    input_chunk_length=1800, output_chunk_length=70, n_epochs=300, random_state=0, add_encoders=encoders, pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": 1
    },
)
model_lgwams.fit(train_amslgw, verbose=False,past_covariates=events)

# COMMAND ----------

pred = model_lgwams.predict(series=train_amslgw, n=70,past_covariates=events)
#pred = model_lgwams.predict(n=70)

val_amslgw.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(val_amslgw, pred)))

# COMMAND ----------

# scale back:
pred_rescaled = amslgw_filled_scaler.inverse_transform(pred)

plt.figure(figsize=(10, 6))
amslgw_filled[-70:].plot(label="actual (air)")
pred_rescaled.plot(label="forecast (air)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multiple series

# COMMAND ----------



# COMMAND ----------


