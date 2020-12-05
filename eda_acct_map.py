"""
Epplorative analysis of the account mapping file

Conclusions:
cust_acct_mapping_id is unique id.
acct_id has 100K unique values
cust_id also has 100K unique values
All roles are primary roles
All src_sys codes are 1
All data_dump_dt are 2017-01-02

summary account mapping file doesn't contain additional informamtion
"""

#%%
import pandas as pd
import os
#%%
acct_map_file = os.getcwd() + "/data/bank_a/accountMapping.csv" 
#%%
# load file
acm = pd.read_csv(acct_map_file)
acm.columns
# %%
acm.head(5)
# %%
acm.describe()
# %%
acm['acct_id'].unique()
# %%
len(acm['cust_id'].unique())
# %%
acm.groupby('data_dump_dt').count()
# %%
