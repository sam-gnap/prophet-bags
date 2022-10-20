# Databricks notebook source
import pandas as pd
from prophet import Prophet

# COMMAND ----------

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`example_wp_log_peyton_manning`;

# COMMAND ----------

df_spark = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`example_wp_log_peyton_manning`')

# COMMAND ----------

df = df_spark.toPandas()

# COMMAND ----------

df.shape

# COMMAND ----------

m = Prophet()
m.fit(df)

# COMMAND ----------

df.ds.max()

# COMMAND ----------

future = m.make_future_dataframe(periods=365)
future.tail()

# COMMAND ----------

forecast = m.predict(future)

# COMMAND ----------

forecast.head()

# COMMAND ----------

fig1 = m.plot(forecast)

# COMMAND ----------

fig2 = m.plot_components(forecast)

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

# COMMAND ----------

plot_components_plotly(m, forecast)

# COMMAND ----------

future_subdaily = m.make_future_dataframe(periods=365, freq='3H')
fcst_subdaily = m.predict(future_subdaily)
fig = m.plot(fcst_subdaily)

# COMMAND ----------

fcst[['ds','yhat']].tail()

# COMMAND ----------

fcst_subdaily[['ds','yhat']].tail()

# COMMAND ----------


