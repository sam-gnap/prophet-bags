# Databricks notebook source
import pandas as pd
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

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

rps = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/RPS_all_sectors.parquet").toPandas()
df_bags = spark.read.parquet("dbfs:/FileStore/shared_uploads/sam.gnap@easyjet.com/sam_b_f.parquet").toPandas()

# COMMAND ----------

df = pd.merge(df_bags,
              rps,
              right_on='flightkey',
              left_on='segment')

# COMMAND ----------

df['rps'] = df['ticketrevenuegbpbudgetnet']/df['lid_y']
df['yield'] = df['ticketrevenuegbpbudgetnet']/df['seatssold']
df['conversions_23'] = df['23kgbags']/df['seatssold']
df['conversions_15'] = df['15kgbags']/df['seatssold']
df['conversions'] = (df['23kgbags']+df['15kgbags'])/df['seatssold']

# COMMAND ----------

df = df[['sector_x','localdepdt','seatssold','rps','yield','conversions_23','conversions_15','conversions']]

# COMMAND ----------

df.head()

# COMMAND ----------

df = df.loc[df['seatssold']>100]

# COMMAND ----------

df = df.groupby(['sector_x', 'localdepdt']).agg({'seatssold': np.sum, 'rps': np.mean, 
                                            'yield': np.mean, 'conversions_23': np.mean, 'conversions_15': np.mean, 
                                            'conversions': np.mean})

# COMMAND ----------

df = df.reset_index(drop=False)

# COMMAND ----------

orynce_multi = df.loc[df['sector_x']=='ORYNCE']

# COMMAND ----------

df = orynce_multi[['localdepdt','rps']].reset_index(drop=True)

# COMMAND ----------

df = df.loc[df['localdepdt']>'2018-01-01']

# COMMAND ----------

# specify dataset information
metadata = MetadataParam(
 time_col="localdepdt",  # name of the time column ("date" in example above)
 value_col="rps",  # name of the value column ("sessions" in example above)
 freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
           # Any format accepted by `pandas.date_range`
)

# COMMAND ----------

events = dict(
    holidays_to_model_separately="auto",
    holiday_lookup_countries="auto",
    holiday_pre_num_days=2,
    holiday_post_num_days=2,
    holiday_pre_post_num_dict=2,
    daily_event_df_dict=2
)

# COMMAND ----------

forecaster = Forecaster()  # Creates forecasts and stores the result
result = forecaster.run_forecast_config(  # result is also stored as `forecaster.forecast_result`.
 df=df,
 config=ForecastConfig(
     model_template='SILVERKITE_DAILY_90',
     forecast_horizon=90,  # forecasts 365 steps ahead
     coverage=0.95,         # 95% prediction intervals
     metadata_param=metadata
 )
)

# COMMAND ----------

grid_search = result.grid_search
cv_results = summarize_grid_search_results(
 grid_search=grid_search,
 decimals=2,
 # The below saves space in the printed output. Remove to show all available metrics and columns.
 cv_report_metrics=None,
 column_order=["rank", "mean_test", "split_test", "mean_train", "split_train", "mean_fit_time", "mean_score_time", "params"])
# Transposes to save space in the printed output
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
cv_results.transpose()

# COMMAND ----------

backtest = result.backtest

# COMMAND ----------

backtest_eval = defaultdict(list)
for metric, value in backtest.train_evaluation.items():
    backtest_eval[metric].append(value)
    backtest_eval[metric].append(backtest.test_evaluation[metric])
metrics = pd.DataFrame(backtest_eval, index=["train", "test"]).T
metrics

# COMMAND ----------

forecast = result.forecast

# COMMAND ----------

fig = forecast.plot_components()

# COMMAND ----------

fig.show()

# COMMAND ----------

summary = result.model[-1].summary()  # -1 retrieves the estimator from the pipeline
print(summary)

# COMMAND ----------

import plotly
from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

# COMMAND ----------

fig = forecaster.plot_quantiles_and_overlays(
 groupby_time_feature="month_dom",
 show_mean=True,
 show_quantiles=False,
 show_overlays=True,
 overlay_label_time_feature="year",
 overlay_style={"line": {"width": 1}, "opacity": 0.5},
 center_values=True,
 xlabel="day of year",
 ylabel=ts.original_value_col,
 title="yearly seasonality for each year (centered)",
)
plotly.io.show(fig)

# COMMAND ----------


