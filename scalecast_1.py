# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster

# COMMAND ----------

df = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`orynce_b_mf`').toPandas()
df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()

# COMMAND ----------

dfm = pd.merge(df_bags[['segment','lid','sold']],
         df,
         on='segment')

# COMMAND ----------

dfm['23kgBagPrice'] = np.exp(dfm['23kgBagPrice_log'])
dfm['15kgBagPrice'] = np.exp(dfm['15kgBagPrice_log'])

# COMMAND ----------

dfm = dfm.loc[dfm['23kgBagPrice']>10]

# COMMAND ----------

dfm.columns = ['segment', 'lid', 'sold', 'DTOG', 'weekdate', 'date_flt', '23kgBags',
       '15kgBags', '23kgBagPrice_log', '15kgBagPrice_log', 'GBPFare_lag',
       'Morning', 'tripduration_days', 'SectorLengthKM', '23kgBagPrice',
       '15kgBagPrice']

# COMMAND ----------

dfm['23conv'] = dfm['23kgBags']/dfm['sold']
dfm['15conv'] = dfm['15kgBags']/dfm['sold']

# COMMAND ----------

dfm.sort_values(by=['segment','DTOG']).head(20)

# COMMAND ----------

dfm['23kgBagPrice'].max(),dfm['23kgBagPrice'].min()

# COMMAND ----------

dfm['15kgBagPrice'].max(),dfm['15kgBagPrice'].min()

# COMMAND ----------

dfm.groupby(by=['DTOG'])[['23conv','15conv']].mean().plot()

# COMMAND ----------

dfm['2315'] = dfm['23kgBagPrice']/dfm['15kgBagPrice']

# COMMAND ----------

dfm.groupby(by=['2315'])[['23conv','15conv']].mean().plot()

# COMMAND ----------

dfm.groupby(by=['DTOG'])[['23kgBagPrice','15kgBagPrice','GBPFare_lag']].mean().plot()

# COMMAND ----------


