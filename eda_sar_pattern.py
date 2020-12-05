"""
Explorative of the SAR pattern.
We will investigate over which time period a SAR covers.
Also design a period indicator 
"""

#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
#%%
trxn_file = os.getcwd() + "/data/bank_a/alert_transactions.csv"
#%%
# load file
trxn = pd.read_csv(trxn_file)
trxn.columns
# %%
y = len(trxn['alert_id'].unique())
print(f"there are {y} unique alert ids")
y = len(trxn.loc[trxn['is_sar']==True]['alert_id'].unique())
print(f"there are {y} unique alert ids related to SARs")
t = trxn['alert_type'].unique()
print(f"there are the following types: {t}")
t = trxn['tx_type'].unique()
print(f"there are the following tx_type: {t}")
r = trxn.groupby('is_sar')['base_amt'].agg([max, min])
print(r)
print(f"transaction amount per label")
# %%
trxn['dag'] = pd.to_datetime(trxn['tran_timestamp'], format='%Y-%m-%d')
r = trxn.groupby(['alert_id', 'is_sar'])['dag'].agg([max, min])
r['delta'] = r['max'] - r['min']
r['delta'].dt.days.plot.hist()

# %%
