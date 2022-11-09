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
from prophet import Prophet

# COMMAND ----------

df = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`example_aa_timeseries`').toPandas()

# COMMAND ----------

df = df.sort_values(by='ds')

# COMMAND ----------

df_train, df_val = df.iloc[:-70], df.iloc[-70:]

# COMMAND ----------

m = Prophet(changepoint_prior_scale=0.001,seasonality_prior_scale=0.01,daily_seasonality=True)

# COMMAND ----------

m.fit(df_train)

# COMMAND ----------

pred = m.predict(df_val)

# COMMAND ----------

pred = pred[['ds','yhat']]

# COMMAND ----------

pred_prophet = pd.merge(pred,
        df_val,
        on='ds')

# COMMAND ----------

pred_prophet

# COMMAND ----------

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

# COMMAND ----------

mape(pred_prophet.y, pred_prophet.yhat)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NBEATSModel

# COMMAND ----------

df = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`example_aa_timeseries`').toPandas()

# COMMAND ----------

df = TimeSeries.from_dataframe(df, time_col='ds',fill_missing_dates=False, freq='d')

# COMMAND ----------

df_filled = utils.missing_values.fill_missing_values(df)
dffilled_scaler = Scaler()
df_filled_scaled = dffilled_scaler.fit_transform(df_filled)
train_df, val_df= df_filled_scaled[:-70], df_filled_scaled[-70:]

# COMMAND ----------

# MAGIC %md
# MAGIC ### One series

# COMMAND ----------

encoders = {"datetime_attribute": {"past": ["dayofweek","month", "year"]}, "cyclic": {"past": ["dayofweek","month","week","day"]}, "transformer": Scaler()}

# COMMAND ----------

model_df = NBEATSModel(
    input_chunk_length=2000, output_chunk_length=70, n_epochs=500, random_state=0, add_encoders=encoders, pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": 1
    },
)
model_df.fit(train_df, verbose=True)

# COMMAND ----------

pred = model_df.predict(n=70)

# COMMAND ----------

val_df.plot(label="actual")
pred.plot(label="forecast")§
plt.legend()

# COMMAND ----------

p = pred.pd_dataframe().reset_index()
v = val_df.pd_dataframe().reset_index()

# COMMAND ----------

mape(v.y, p.y)

# COMMAND ----------


