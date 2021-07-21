#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os
import sys
GRAPHCASE_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
sys.path.insert(0, GRAPHCASE_FOLDER)
import tensorflow as tf
from GAE.graph_case_controller import GraphAutoEncoder
from AMLSim_preproccessor import AmlSimPreprocessor

class ReconInsight:

    def __init__(self, trxn_file, acct_file, alert_file, out_dir, layers):
        self.pp = AmlSimPreprocessor(trxn_file, acct_file, alert_file, out_dir, layers)
        
    def create_insight(self, dim_size, id):
        gae = self.pp.train_model(dim_size)
        embeds = self.create_embedding(gae, 1)
        # embed = gae.get_l1_structure(id, graph=None, verbose=None, show_graph=False,
        #                  node_label=None, get_pyvis=False)
        print(embeds.shape)
        

    def create_embedding(self, gae, dag):
        node, edge = self.pp.proces_month(dag)
        G = self.pp.create_graph(node, edge)
        return gae.calculate_embeddings(G)

#%%

trxn_file = os.getcwd() + "/data/bank_a/transactions.csv"
acct_file = os.getcwd() + "/data/bank_a/accounts.csv" 
alert_file = os.getcwd() + "data/bank_a/alert_transactions.csv"
out_dir = os.getcwd() + "/data/bank_a/"

ri = ReconInsight(trxn_file, acct_file, alert_file, out_dir, 4)
ri.pp.epochs = 10
ri.pp.dropout_levels = [0]
ri.pp.act_functions = [tf.nn.sigmoid]

ri.create_insight(32, 1)
# %%
