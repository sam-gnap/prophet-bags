# Databricks notebook source
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
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
torch.manual_seed(1)
np.random.seed(1)

# COMMAND ----------

series_air = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`airpassengers`').toPandas()
series_milk = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`milkproduction`').toPandas()

# COMMAND ----------

series_air.columns = ['date','air']
series_milk.columns = ['date','milk']

# COMMAND ----------

a = TimeSeries.from_dataframe(series_air, time_col='date')
s = TimeSeries.from_dataframe(series_milk, time_col='date')

# COMMAND ----------

a.plot(label="Number of air passengers")
s.plot(label="Pounds of milk produced per cow")
plt.legend()

# COMMAND ----------

scaler_air, scaler_milk = Scaler(), Scaler()
series_air_scaled = scaler_air.fit_transform(a)
series_milk_scaled = scaler_milk.fit_transform(s)

series_air_scaled.plot(label="air")
series_milk_scaled.plot(label="milk")
plt.legend()

# COMMAND ----------

train_air, val_air = series_air_scaled[:-36], series_air_scaled[-36:]
train_milk, val_milk = series_milk_scaled[:-36], series_milk_scaled[-36:]

# COMMAND ----------

model_air = NBEATSModel(
    input_chunk_length=24, output_chunk_length=12, n_epochs=200, random_state=0, pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": 1
    },
)

# COMMAND ----------

model_air.fit(train_air, verbose=False)

# COMMAND ----------

pred = model_air.predict(n=36)

series_air_scaled.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(series_air_scaled, pred)))

# COMMAND ----------


