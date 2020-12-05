"""
Explorative analysis of the transactions file

Ongeveer 10mln trxn, of type transfer over period jan 2017 to dec 2018
outdegree is alway between 102 to 104
indegree ranges between zero and 1137
The amounts are between 0 en 3000, with a peak around 500 USD.
Note that the amounts related to sar are all above the 1600Eur
"""

#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%%
trxn_file = os.getcwd() + "/data/bank_a/transactions.csv"
#%%
# load file
trxn = pd.read_csv(trxn_file)
trxn.columns
# %%
trxn.describe()
# %%
trxn.groupby('tx_type')['tran_id'].count()
#%%
#check over which period
trxn['dag'] = pd.to_datetime(trxn['tran_timestamp'], format='%Y-%m-%d')
trxn.groupby([trxn["dag"].dt.year, trxn["dag"].dt.month])['tran_id'].count().plot(kind="bar")
plt.title('trxn by period')
plt.show

# %%
# check outdegree degree distri
trxn.groupby('orig_acct')['tran_id'].count().hist()
plt.title('outdegree')
plt.show
# %%
# check incoming degree distri
trxn.groupby('bene_acct')['tran_id'].count().hist()
plt.title('indegree')
plt.show
# %%
#SAR distri
trxn.loc[trxn['is_sar']==True].groupby([trxn["dag"].dt.year, trxn["dag"].dt.month])['tran_id'].count().plot(kind="bar")
plt.title('sar by period')
plt.show
# %%
# check the amount distribution
trxn['log_amount'] = np.log10(trxn['base_amt'])
trxn['log_amount'].hist()
# %%
trxn['base_amt'].hist()

# %%
tmp = trxn[['is_sar', 'base_amt']]
tmp.groupby('is_sar').plot.hist(alpha=0.5)
# %%
tmp.loc[tmp['is_sar']==True].describe()
#%%
tmp.loc[tmp['is_sar']==False].describe()
# %%
len(trxn.loc[trxn['is_sar']==True]['alert_id'].unique())
# %%
trxn.loc[trxn['is_sar']==True]['alert_id'].unique()

# %%
