# Databricks notebook source
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts import utils
from darts.models import NBEATSModel
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# COMMAND ----------

df = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()

# COMMAND ----------

df = df.loc[df['lid']>100]
df = df.loc[df['localdepdt']>'2015-10-01']
df['rps'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']

# COMMAND ----------

lgw = df.loc[df['sector'].str[:3]=='LGW',['sector','localdepdt','rps']]

# COMMAND ----------

lgw = lgw.loc[lgw['localdepdt']>'2021-05-01'].reset_index()

# COMMAND ----------

lgw = lgw.groupby(by=['sector','localdepdt'])['rps'].mean().reset_index()

# COMMAND ----------

selected_sectors = (lgw.groupby('sector')['rps'].count()>360).reset_index()

# COMMAND ----------

lgw = lgw.loc[lgw.sector.isin(selected_sectors.loc[selected_sectors['rps']==True].sector)]

# COMMAND ----------

t = TimeSeries.from_group_dataframe(lgw, group_cols='sector',time_col='localdepdt',fill_missing_dates=True, freq='d',fillna_value=True)

# COMMAND ----------

t_scaler = Scaler()
t_scaled = t_scaler.fit_transform(t)

# COMMAND ----------

encoders = {"datetime_attribute": {"past": ["dayofweek","month", "year"]}, "cyclic": {"past": ["dayofweek","month","week","day"]}, "transformer": Scaler()}

# COMMAND ----------

model_lgwams = NBEATSModel(
    input_chunk_length=300, output_chunk_length=70, n_epochs=300, random_state=0, add_encoders=encoders, pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": 1
    },
)
model_lgwams.fit(t, verbose=True)

# COMMAND ----------

pred = model_lgwams.predict(series=train_lgwams, n=70)
val_lgwams.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(val_lgwams, pred)))

# COMMAND ----------


