"""
Class for pre-processing the transactions and account files into a node and edge set that can be
used for creating a networkx graph. The preprocessing contains the following steps:

1)  select trxn for period
2) item check id nummering
3) item create features and normalize
4) item save node and edge data. Node date has the fields ['id', 'orig_id', 'month', features]
    edge data has the fields, orig, benef, amount

@param

input: file name of the transaction and account, save location, month.
"""
#%%

import os
import sys
import pickle
import pandas as pd
import networkx as nx
import tensorflow as tf
from datetime import datetime

# GRAPHCASE_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
# sys.path.insert(0, GRAPHCASE_FOLDER)
from GAE.graph_case_controller import GraphAutoEncoder
from xg_gridsearch import XgGridSearch


class AmlSimPreprocessor:
    learning_rates = [0.001, 0.005, 0.0005]
    dropout_levels = [0, 1, 2]
    act_functions = [tf.nn.tanh, tf.nn.sigmoid]
    dim_size = 32
    epochs = 5000

    def __init__(self, trxn_file, acct_file, alert_file, out_dir, layers):
        self.trxn_file = trxn_file
        self.acct_file = acct_file
        self.alert_file = alert_file
        self.out_dir = out_dir
        self.max_series = None
        self.amt_max = None
        self.layers = layers

    def proces_month(self, dag):
        trxn = self.get_wire_data(dag)
        feat = self.create_features2(trxn, dag)
        feat = self.normalize_features(feat)
        feat = self.mergeLabels(feat, dag)
        feat = self.reset_index(feat)
        edge = self.retrieve_edges(trxn, feat, dag)
        edge = self.normalize_edges(edge)
        self.save_dfs(feat,edge, dag)
        return feat, edge

    def get_wire_data(self, dag):
        # select relevant periods
        trxn = pd.read_csv(self.trxn_file)
        trxn['trxn_dt'] = pd.to_datetime(trxn['tran_timestamp'], format='%Y-%m-%d')
        trxn['dag'] = (trxn['trxn_dt'].dt.year - 2017) * 12 + trxn['trxn_dt'].dt.month
        trxn = trxn.loc[(trxn['dag']==dag) | (trxn['dag']==(dag-1))]
        return trxn

    def create_features(self, trxn, dag):
        """
        Creates the in and outgoing features.
        """

        def determine_periode(row, dag, dir='inc'):
            row[dir + '_first_half'] = 0
            row[dir + '_second_half'] = 0
            row[dir + '_prior_month'] = 0
            row[dir + '_cnt'] = 1

            if row['dag']==dag:
                if row['trxn_dt'].day<=15:
                    row[dir + '_first_half'] = row['base_amt']
                else:
                    row[dir + '_second_half'] = row['base_amt']
            else:
                row[dir + '_prior_month'] = row['base_amt']
            return row

        feat = None 
        for dir, fld in [('inc', 'bene_acct'), ('outg', 'orig_acct')]:
            trxn = trxn.apply(determine_periode, dag=dag, dir=dir, axis=1)
            res = trxn.groupby(fld)[dir + '_first_half', dir + '_second_half', dir + '_prior_month', dir + '_cnt'].sum()
            if feat is None:
                feat = res
            else:
                feat = feat.merge(res, left_index=True, right_index=True, how='outer')

            feat = feat.fillna(0)
        return feat

    def qa_check(self,dag):
        trxn = self.get_wire_data(dag)
        feat = self.create_features2(trxn, dag)
        print(f"max transacion amount: {self.amt_max}")
        print(f"feat_max: {self.max_series}")
        print(feat.describe())
        feat = self.normalize_features(feat)
        feat = self.reset_index(feat)
        edge = self.retrieve_edges(trxn, feat, dag)
        print(edge.describe())

    def normalize_features(self, df):
        # min-max normalize values
        if self.max_series is None:
            self.max_series = df.max()

        df_norm = df/self.max_series
        df_norm = df_norm.fillna(0)
        return df_norm

    def normalize_edges(self,df):
        if self.amt_max is None:
            self.amt_max = df['weight'].max()

        df['weight'] = df['weight'] / self.amt_max
        df = df.fillna(0)
        return df

    def reset_index(self, feat):
        # # reset id starting from 0
        feat['orig_id'] = feat.index
        feat.reset_index(inplace=True, drop=True)
        feat['id'] = feat.index 
        return feat

    def retrieve_edges(self, trxn, feat, dag):
        print(trxn.columns)
        edge = trxn.loc[trxn['dag']==dag][['dag', 'orig_acct', 'bene_acct', 'base_amt']]
        edge.rename(columns={'base_amt':'weight'}, inplace=True)

        # mapping
        node_mapping = feat[['id', 'orig_id']]

        # join mapping with the orig.
        edge = pd.merge(edge, node_mapping, left_on='orig_acct', right_on='orig_id', how='left')
        edge.rename(columns={'id':'source'}, inplace=True)
        edge.drop(labels=['orig_acct', 'orig_id'], axis=1, inplace=True)

        # benef remapping
        edge = pd.merge(edge, node_mapping, left_on='bene_acct', right_on='orig_id', how='left')
        edge.rename(columns={'id':'target'}, inplace=True)
        edge.drop(labels=['bene_acct', 'orig_id'], axis=1, inplace=True)
        return edge

    def save_dfs(self, node, edge, sar):
        node.to_parquet(self.out_dir + 'node_' + str(sar))
        edge.to_parquet(self.out_dir + 'edge_' + str(sar))

    def create_features2(self, trxn, dag): 
        # create indicators for aggregation
        trxn.loc[trxn['dag']==(dag - 1), 'prior_month'] = trxn['base_amt']
        trxn.loc[(trxn['dag']==(dag)) & (trxn['trxn_dt'].dt.day<=15), 'first_half'] = trxn['base_amt']
        trxn.loc[(trxn['dag']==(dag)) & (trxn['trxn_dt'].dt.day>15), 'second_half'] = trxn['base_amt']
        trxn['cnt'] = 1
        feat = None 

        for fld in ['bene_acct', 'orig_acct']:
            res = trxn.groupby(fld)['first_half', 'second_half', 'prior_month', 'cnt'].sum()
            if feat is None:
                feat = res
            else:
                feat = feat.merge(res, left_index=True, right_index=True, how='outer',
                                  suffixes=['_in', '_out'])

            feat = feat.fillna(0)
        return feat
    
    def check_node(self, node, node_id = 2):
        trxn = self.get_wire_data(node_id)
        trxn_out = trxn.loc[trxn['orig_acct']==node_id]
        print("out")
        print(trxn_out['base_amt'].agg(['sum', 'count']))
        r = node.loc[node['orig_id']==2][['first_half_out', 'second_half_out', 'prior_month_out', 'cnt_out']]
        m = self.max_series[['first_half_out', 'second_half_out', 'prior_month_out', 'cnt_out']]
        r = r * m
        print(r['first_half_out'] + r['second_half_out'] + r['prior_month_out'])
        print(r['cnt_out'])

        print("in")
        trxn = trxn.loc[trxn['bene_acct']==node_id]
        print(trxn['base_amt'].agg(['sum', 'count']))
        r = node.loc[node['orig_id']==2][['first_half_in', 'second_half_in', 'prior_month_in', 'cnt_in']]
        m = self.max_series[['first_half_in', 'second_half_in', 'prior_month_in', 'cnt_in']]
        r = r * m
        print(r['first_half_in'] + r['second_half_in'] + r['prior_month_in'])
        print(r['cnt_in'])

    def create_graph(self, nodes, edge, field_list=None):
        '''
        creates a networkx graph of the elliptics dataset
        refer to Elliptic, www.elliptic.co.
        '''
        if field_list:
            nodes = nodes[field_list] 

        G = nx.from_pandas_edgelist(edge, 'source', 'target', ["weight"], create_using=nx.DiGraph)

        for col in nodes.columns:
            if col not in ['id', 'orig_id', 'is_sar', 'dag']:
                nx.set_node_attributes(G, pd.Series(nodes[col], index=nodes.id).to_dict(), col)
        return G
    
    def gs_graphcase(self, G, dim_size):
        gs_res = {}
        if dim_size > 8:
            dims = [6] + [dim_size] * (self.layers -1)
        else:
            dims = [dim_size] * self.layers
        gae = GraphAutoEncoder(G, learning_rate=0.001, support_size=[5, 5], dims=dims,
                               batch_size=1024, max_total_steps=AmlSimPreprocessor.epochs, verbose=False, act=tf.nn.sigmoid)

        for lr in AmlSimPreprocessor.learning_rates:
            for do in AmlSimPreprocessor.dropout_levels:
                for act in AmlSimPreprocessor.act_functions:
                    train_res = {}
                    for i in range(len(gae.dims)):
                        if i in range(1, 1 + do):
                            train_res["l"+str(i+1)] = gae.train_layer(i+1, dropout=0.1, learning_rate=lr, act=act)
                        else:
                            train_res["l"+str(i+1)] = gae.train_layer(i+1, dropout=None, learning_rate=lr, act=act)

                    train_res['all'] = gae.train_layer(len(gae.dims), all_layers=True, learning_rate=lr, act=act)
                    
                    # save results
                    act_str =  'tanh' if act==tf.nn.tanh else 'sigm'
                    run_id = f'dim_{dim_size}_lr_{lr}_do_{do}_act_{act_str}_layers_{self.layers}'
                    pickle.dump(train_res, open(self.out_dir + 'res_' + run_id, "wb"))
                    gae.save_model(self.out_dir + 'mdl_' + run_id)

                    # print and store result
                    val_los = sum(train_res['all']['val_l'][-4:]) / 4
                    gs_res[run_id] = val_los
                    print(f'dims:{dim_size}, lr:{lr}, dropout lvl:{do}, act func:{act_str} resultsing val loss {val_los}')

        # print all results, save and return best model
        for k, v in gs_res.items():
            print(f'run: {k} with result {v}')
        pickle.dump(gs_res, open(self.out_dir + f'graphcase_gs_results_dim_{dim_size}', "wb"))
        return max(gs_res, key=gs_res.get)
        
    def create_embedding(self, mdl, dim_size):
        gae = None
        combined_feat = None
        for dag in range(1,25):
            print(f"processing dag {dag}")
            node, edge = self.proces_month(dag)
            cnt = node.shape[0]
            G = self.create_graph(node, edge)
            if gae is None:
                dims = [int(mdl.split("_")[1])] * self.layers
                act = tf.nn.sigmoid if mdl.split("_")[7]=='sigm' else tf.nn.tanh
                gae = GraphAutoEncoder(G, learning_rate=0.001, support_size=[5, 5], dims=dims,
                               batch_size=1024, max_total_steps=1000, verbose=False, act=act)
                gae.load_model(self.out_dir + 'mdl_' + mdl, G)
            embed = gae.calculate_embeddings(G)

            #combine with nodes
            if self.layers % 2 == 0:
                pd_embed = pd.DataFrame(data=embed[:cnt,1:], index=embed[:cnt,0], columns=[f'embed_{i}' for i in range(dims[-1] * 2)])
            else:
                pd_embed = pd.DataFrame(data=embed[:cnt,1:], index=embed[:cnt,0], columns=[f'embed_{i}' for i in range(dims[-1])])
            feat = pd.merge(node, pd_embed, left_index=True, right_index=True, how='inner')

            if combined_feat is None:
                combined_feat = feat
            else:
                combined_feat = pd.concat([combined_feat, feat])
        
        feat_file = self.out_dir + "features_" + str(dim_size)
        combined_feat.to_parquet(feat_file)

        # return column list
        excl_cols = ['is_sar', 'dag', 'orig_id', 'id']
        feat_cols = [c for c in combined_feat.columns if c not in excl_cols]
        return feat_file, feat_cols

    def get_labels(self):
        #load alerts trxn and filter on sar
        alerts = pd.read_csv(self.trxn_file)
        sars = alerts.loc[alerts['is_sar']==True]

        # get the highest dag per sar
        sars['trxn_dt'] = pd.to_datetime(sars['tran_timestamp'], format='%Y-%m-%d')
        sars['dag'] = (sars['trxn_dt'].dt.year - 2017) * 12 + sars['trxn_dt'].dt.month
        sar_hdr = sars.groupby('alert_id')['dag'].agg([max])
        sar_hdr.rename(columns={'max':'dag'}, inplace=True)

        #join dag with SAR trxn
        label = sars[['alert_id', 'orig_acct', 'bene_acct']]
        label = label.merge(sar_hdr, left_on='alert_id', right_index=True, how='inner')
        lbl_orig = label[['orig_acct', 'dag']]
        lbl_orig.rename(columns={'orig_acct':'acct'}, inplace=True)
        lbl_bene = label[['bene_acct', 'dag']]
        lbl_bene.rename(columns={'bene_acct':'acct'}, inplace=True)
        lbls = pd.concat([lbl_orig, lbl_bene])
        lbls.drop_duplicates(subset=['acct', 'dag'], inplace=True, ignore_index=True)
        lbls['is_sar'] = 1
        return lbls

    def mergeLabels(self, feat, dag):
        lbls = self.get_labels()
        lbls = lbls.loc[lbls.dag==dag]
        feat['acct'] = feat.index
        feat = feat.merge(lbls, how='left', on='acct')
        feat.drop(['acct', 'dag'], axis=1, inplace=True)
        feat.fillna(0, inplace=True)
        feat['dag'] = dag
        return feat

    def controller(self, dim_size):
        # create node and edge df
        node, edge = self.proces_month(2)
        G = self.create_graph(node, edge)
        mdl = self.gs_graphcase(G, dim_size)
        print(mdl)  # dim_4_lr_0.0005_do_2_act_sigm_layers_5
        feat_file, feat_cols = self.create_embedding(mdl, dim_size)
        splits = [10 ,17 , 26]
        lbl_name = 'is_sar'
        mdl_id = "dim_size_" + str(self.dim_size)
        gs = XgGridSearch(feat_file, splits, feat_cols, lbl_name, self.out_dir, mdl_id)
        res = gs.controller(verbose=False)
        res['graphcase_model'] = mdl
        return res

    def dim_size_search(self, dim_sizes):
        res = []
        for s in dim_sizes:
            res.append(self.controller(s))
        df = pd.DataFrame(res)
        df.to_parquet(self.out_dir + "dim_size_gs_res")
        return df

    def controller_without_embed(self, feat_file):
        # feat_file = self.out_dir + "features_4"
        feat_cols = ['first_half_in', 'second_half_in', 'prior_month_in', 'cnt_in',
            'first_half_out', 'second_half_out', 'prior_month_out', 'cnt_out']
        splits = [10 ,17 , 26]
        lbl_name = 'is_sar'
        mdl_id = "dim_size_0"
        gs = XgGridSearch(feat_file, splits, feat_cols, lbl_name, self.out_dir, mdl_id)
        res = gs.controller(verbose=False)
        res['graphcase_model'] = mdl_id
        return res


        

# trxn_file = os.getcwd() + "/data/bank_a/transactions.csv"
# acct_file = os.getcwd() + "/data/bank_a/accounts.csv" 
# alert_file = os.getcwd() + "data/bank_a/alert_transactions.csv"
# out_dir = os.getcwd() + "/data/bank_a/"

# pp = AmlSimPreprocessor(trxn_file, acct_file, alert_file, out_dir, 4)


# # %%
# pp.controller(4)
# #%%
# node, edge = pp.proces_month(2)
# G = pp.create_graph(node, edge)
# mdl = pp.gs_graphcase(G, 4)
# print(mdl)  # dim_4_lr_0.0005_do_2_act_sigm_layers_5
# feat = pp.create_embedding(mdl)
# # edge.head(4)
# # %%
# pp.qa_check(2)
# # %%
# pp.check_node(node, 2)