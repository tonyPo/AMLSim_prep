"""
Explorative analysis of the alerts and sar file

conclusions alert_file::
There are 200 alerts related to 890 accounts
Model_id is 3 for all alerts
all alerts are of type cycle. Cycle have length between 3 and 6.
no clear correlation between cycle length and is_sar indicator.
of the 890 alerts, 440 account in 100 SARs and 450 false positives.
start en end is respectively 0 and 1mln.
schedule id = 0
bank_id = bank_a 
only one alert per account

Sar_file:
440 SARs of type cycle, IS_SAR flag set to yes and with ACCOUNT_TYPE set to INDIVIDUAL
The SAR are spread between 2017-01 and 2019-12, last month has less cases.
Note that every account can only have one alert /sar. I.e. alert generation/SAr filing is not 
per month.
"""
#%%

import pandas as pd
import os
#%%
alert_file = os.getcwd() + "/data/bank_a/alert_accounts.csv"
sar_file = os.getcwd() + "/data/bank_a/sar_accounts.csv"
#%%
# load file
alert = pd.read_csv(alert_file)
alert.columns
# %%
alert.describe()
# %%
alert.groupby('alert_type').count()
# %%
alert.groupby('is_sar').count()
# %%
sar = pd.read_csv(sar_file)
sar.columns
# %%
sar.describe()
# %%
sar.groupby('ACCOUNT_TYPE').count()
# %%
sar['dag'] = pd.to_datetime(sar['EVENT_DATE'], format='%Y%m%d')

# sar['dag'].plot.hist(13)
# %%
sar['dag']  
# %%
sar.groupby([sar["dag"].dt.year, sar["dag"].dt.month])['ALERT_ID'].count().plot(kind="bar")
# %%
sar.groupby('ALERT_ID').agg({'dag': [min, max], 'MAIN_ACCOUNT_ID': 'count' })
# %%
tmp = sar.merge(alert.loc[alert['is_sar']==True], 'inner', left_on='MAIN_ACCOUNT_ID', right_on='acct_id')
tmp.shape
# %%
# check distrubution of # of nodes per alert.
tmp = alert.groupby(['alert_id', 'is_sar'])['acct_id'].count()
tmp
# %%
tmp.plot.hist(by='is_sar')
# %%
tmp.groupby('is_sar').plot.hist(alpha=0.5)
# %%
