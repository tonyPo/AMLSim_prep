# AMLSim_prep
EDA of the AMLSim synthetic dataset, feature and graph creation

# EDA analysis

## accounts
Epplorative analysis of the accounts file

Conclusions:
- The following columns are not populated and therefore can be removed: 'first_name','last_name', 'street_addr', 'city', 'state', 'country', 'zip', 'gender', 'birth_date', 'ssn', 'lon', 'lat']
- dsply_nm and acct_id are a one-to-one mapping
- All accounts have type SAV
- All account have status A(ctive)
- All accounts have USD as reporting currency
- allw account have prior_sar_count = True
- All account have the same branch id
- All acount have the same open_dt and close_dt
- initial deposit is between 50K and 100K
- All acount have the same tx_behavior_id and bank_a

summary:
Only the fields acct_id, prior_sar_count, initial deposit contain usefull information.


## Account mapping
Epplorative analysis of the account mapping file

Conclusions:
- cust_acct_mapping_id is unique id.
- acct_id has 100K unique values
- cust_id also has 100K unique values
- All roles are primary roles
- All src_sys codes are 1
- All data_dump_dt are 2017-01-02

summary account mapping file doesn't contain additional informamtion

## alerts
There is an alert file and SAR file. Note that the alert file contains false positives.
false positives are pattern that look like SARs (true positives) but deviate in high of amounts and timing.

Conclusions alert_file:
- There are 200 alerts related to 890 accounts
- Model_id is 3 for all alerts
- All alerts are of type cycle. Cycle have length between 3 and 6.
- No clear correlation between cycle length and is_sar indicator.
- Of the 890 alerts, 440 account in 100 SARs and 450 false positives.
- Start en end is respectively 0 and 1mln.
- schedule id = 0
- bank_id = bank_a 


Sar_file:
- 440 SARs of type cycle, IS_SAR flag set to yes and with ACCOUNT_TYPE set to INDIVIDUAL
- The SARs are spread between 2017-01 and 2019-12, last month has less cases.

Note that every account can only have one alert /sar. I.e. alert generation/SAr filing is not 
per month.

alert_transactions file:
- Alerts are all cycles using transactions of type transfer.
- 100 SARs and 100 false posive sars.
- **Amount of SAR is significant higher (>1600 USD) compared to false positve alerts (<200USD)**
- **Cycles for SARs take place in a time span of max 30 days and for false postivite between 30 and 90 days.**


## transactions
- Around 10mln trxns of type transfer over period Jan 2017 to Dec 2018
- **Outdegree is alway between 102 to 104** this is not a realistic assumption.
- Indegree ranges between 0 and 1137
- The amounts are between 0 en 3000, with a peak around 500 USD.
- Note that the amounts related to sar are all above the 1600Eur!
