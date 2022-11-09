# Databricks notebook source
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score

# COMMAND ----------

def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(8, 5))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(
        "R2: {}".format(r2_score(ts_transformed.univariate_component(0), pred_series))
    )
    plt.legend()

# COMMAND ----------

df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()
df_events.columns = list(map(str.lower,df_events.columns))
df = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()
df.columns = list(map(str.lower,df.columns))

# COMMAND ----------

df['rps'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']
df = df.loc[df['lid']>100]
df = df.reset_index()

# COMMAND ----------

df.head(2)

# COMMAND ----------

df_ORYNCE=df.loc[df['sector']=='ORYNCE']

# COMMAND ----------

df_ORYNCE = df_ORYNCE.groupby(by=['localdepdt'])['rps'].mean().reset_index()
df_ORYNCE = df_ORYNCE.reset_index(drop=True)
df_ORYNCE.columns = ['ds','y']

# COMMAND ----------

df_ORYNCE

# COMMAND ----------

series = TimeSeries.from_dataframe(df_ORYNCE, "ds", "y",freq='d')
train, val = series[:-60], series[-60:]

# COMMAND ----------

filler = MissingValuesFiller()
scaler = Scaler()
series = scaler.fit_transform(
    filler.transform(
        TimeSeries.from_dataframe(
            df_ORYNCE, "ds", ["y"], freq='d'
        )
    )
).astype(np.float32)
series.plot()

# COMMAND ----------

train, val = series.split_after(pd.Timestamp("20220801"))

# COMMAND ----------

encoders = {"datetime_attribute": {"past": ["dayofweek","month", "year"]}, "transformer": Scaler()}

# COMMAND ----------

encoders

# COMMAND ----------

model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    add_encoders=encoders,
    random_state=42,
)

# COMMAND ----------

model.fit(train, epochs=20, verbose=True);

# COMMAND ----------

pred_air = model.predict(series=val, n=36)

# scale back:
pred_air = scaler.inverse_transform(pred_air)

# COMMAND ----------

plt.figure(figsize=(10, 6))
series_air.plot(label="actual (air)")
pred_air.plot(label="forecast (air)")

# COMMAND ----------

import  matplotlib.pyplot as plt

# COMMAND ----------

pred_series = model.historical_forecasts(
    series,
    start=pd.Timestamp("20220801"),
    forecast_horizon=1,
    stride=5,
    retrain=False,
    verbose=True,
)
display_forecast(pred_series, series, "1 day", start_date=pd.Timestamp("20220801"))

# COMMAND ----------

display_forecast(pred_series, series, "1 day", start_date=pd.Timestamp("20220801"))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

m = Prophet(holidays=events,changepoint_prior_scale=0.001,seasonality_prior_scale=0.01)

# COMMAND ----------

m.fit(data)

# COMMAND ----------

df_cv = cross_validation(m, initial='730 days', period='60 days', horizon = '180 days', parallel="processes")
df_p = performance_metrics(df_cv)

# COMMAND ----------

df_p.mdape.mean(), df_p.smape.mean()

# COMMAND ----------

prediction = m.predict(data.loc[data['ds']>'2022-08-01',['ds']])

# COMMAND ----------

df_merged = pd.merge(prediction[['ds','yhat']],
         data.loc[data['ds']>'2022-08-01',['ds','y']],
        on='ds')

# COMMAND ----------

fig1 = m.plot(prediction)
fig2 = m.plot_components(prediction)
