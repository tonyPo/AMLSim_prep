"""
Epplorative analysis of the accounts file

Conclusions:
The following columns are not populated and therefore can be removed
'first_name','last_name', 'street_addr', 'city', 'state', 'country', 'zip', 'gender',
'birth_date', 'ssn', 'lon', 'lat']
dsply_nm and acct_id are a one-to-one mapping
All accounts have type SAV
All account have status A(ctive)
All accounts have USD as reporting currency
allw account have prior_sar_count = True
All account have the same branch id
All acount have the same open_dt and close_dt
initial deposit is between 50K and 100K
All acount have the same tx_behavior_id and bank_a

summary:
Only the fields acct_id, prior SAR count, initial deposit contain usefull information.

"""
#%%
import pandas as pd
import os
#%%
acct_file = os.getcwd() + "/data/bank_a/accounts.csv" 
#%%
# load file
acct = pd.read_csv(acct_file)
acct.columns
# %%
acct.describe()
acct.groupby('acct_stat').count()
acct.groupby('type').count()
#...
acct['initial_deposit'].plot.hist(bins=20)

# %%

