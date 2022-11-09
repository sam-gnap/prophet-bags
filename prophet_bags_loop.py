# Databricks notebook source
import pandas as pd
import numpy as np
from prophet import Prophet 
pd.set_option('display.max_rows', 100)

# COMMAND ----------

bags1 = spark.sql('select * from `ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_1`').toPandas()
bags2 = spark.sql('select * from `ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_2`').toPandas()
bags3 = spark.sql('select * from `ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_3`').toPandas()

# COMMAND ----------

df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()

# COMMAND ----------

predictions = pd.concat([bags1,bags2,bags3])

# COMMAND ----------

predictions.head()

# COMMAND ----------

df_bags_processes = pre_process_bags(df_bags)

# COMMAND ----------

df = pd.concat([df_bags_processes,predictions])

# COMMAND ----------

outname = 'pre-processed.csv'
outdir = '/dbfs/FileStore/'
dfPandas.to_csv(outdir+outname, index=False, encoding="utf-8")

# COMMAND ----------

df.to_parquet("/dbfs/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_pred.parquet")

# COMMAND ----------

df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()
df_future = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/future_f.parquet").toPandas()
df_events = sqlContext.sql('SELECT * FROM `ds_data_analytics`.`data_analytics_sandbox`.`sam_e_all`').toPandas()

# COMMAND ----------

def process_holidays(df_events):
    df_events['ds'] = pd.to_datetime((df_events['StartDate'] + (df_events['EndDate']-df_events['StartDate'])/2).dt.date)
    df_events['lower_window'] = (df_events['StartDate']-df_events['ds']).dt.days
    df_events['upper_window'] = np.abs((df_events['EndDate']-df_events['ds'])).dt.days
    df_events = df_events[['Name','ds','lower_window','upper_window','Origin','Destination']]
    df_events.columns = ['holiday','ds','lower_window','upper_window','origin','destination']
    lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2021-05-01','origin':None,'destination':None},])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    df_events = pd.concat([df_events,lockdowns[['holiday', 'ds', 'lower_window', 'upper_window','origin','destination']]]).reset_index(drop=True)
    return df_events

# COMMAND ----------

def pre_process_bags(df):
    df = df.loc[df['lid']>100]
    df = df.loc[df['sold']>20]
    df['bags'] = df['23kgbags']+df['15kgbags']
    df['bags_conversions'] = df['bags']/df['sold']
    df = df[['sector','std','bags_conversions']]
    df.loc[df['bags_conversions']>1,'bags_conversions']=1
    df.columns = ['sector','ds','y']
    return df

# COMMAND ----------

def future_pre_process(df):
    df['sector'] = df['Series'].str[:6]
    df = df[['sector','STD']]
    df.columns = ['sector','ds']
    return df

# COMMAND ----------

df_bags = pre_process_bags(df_bags)
df_future = future_pre_process(df_future)
df_events = process_holidays(df_events)

# COMMAND ----------

df_bags.ds.max()

# COMMAND ----------

df_future = df_future.loc[df_future['ds']>'2022-10-14']

# COMMAND ----------

sectors_500 = (df_bags.groupby(by='sector')['ds'].count()>500).reset_index()
sector_500_unique = sectors_500.loc[sectors_500['ds']==True].sector

# COMMAND ----------

df_bags = df_bags.loc[df_bags['sector'].isin(sector_500_unique)]

# COMMAND ----------

df_bags.sector.nunique()

# COMMAND ----------

temp_df = pd.DataFrame()
for i, sector in enumerate(df_bags.sector.unique()):
    print('--------------------------')
    print('--------------------------')
    print(i,sector)
    if i==300:
        spark_df = spark.createDataFrame(temp_df)
        spark_df.write.mode("overwrite").saveAsTable("`ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_all_1`")
        temp_df = pd.DataFrame()
    elif i==600:
        spark_df = spark.createDataFrame(temp_df)
        spark_df.write.mode("overwrite").saveAsTable("`ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_all_2`")
        temp_df = pd.DataFrame()
    elif i==900:
        spark_df = spark.createDataFrame(temp_df)
        spark_df.write.mode("overwrite").saveAsTable("`ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_all_3`")
        temp_df = pd.DataFrame()
    else:
        df_future_sector = df_future.loc[df_future['sector']==sector,['ds']]
        if df_future_sector.empty:
            print(f'No future flights for {sector}')
            continue
        df_history = df_bags.loc[df_bags['sector']==sector,['ds','y']]
        relevant_events = df_events.loc[(df_events['origin']==sector[:3])|(df_events['destination']==sector[3:])\
                                      |(df_events['origin'].isnull())&(df_events['destination'].isnull())
                                      ,['holiday','ds','lower_window','upper_window']].reset_index(drop=True)
        m = Prophet(holidays=relevant_events,
                    changepoint_prior_scale=0.001,
                    seasonality_prior_scale=0.01,
                    seasonality_mode='multiplicative')
        m.fit(df_history)
        prediction = m.predict(df_future_sector)
        prediction = prediction[['ds','trend','yhat','holidays','weekly','yearly']]
        prediction['sector'] = sector
        temp_df = pd.concat([temp_df,prediction])

# COMMAND ----------

spark_df = spark.createDataFrame(temp_df)

spark_df.write.mode("overwrite").saveAsTable("`ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_all_1`")

# COMMAND ----------

pd_df = spark.sql('select * from `ds_data_analytics`.`data_analytics_sandbox`.`bags_pred_3`').toPandas()

# COMMAND ----------

pd_df.head()

# COMMAND ----------


