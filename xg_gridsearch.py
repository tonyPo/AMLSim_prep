#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import random
from sklearn import metrics
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class XgGridSearch:
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'max_delta_step': hp.quniform('max_delta_step', 1, 10, 1),
        'learning_rate': hp.quniform('learning_rate', 0.025, 0.1, 0.025),
        'reg_lambda': hp.choice('reg_lambda', np.arange(0, 2, dtype=int)),
        'reg_alpha': hp.choice('reg_alpha', np.arange(0, 2, dtype=int)),
        'scale_pos_weight': hp.quniform('scale_pos_weight', 1, 100, 1),
        'gamma': hp.quniform('gamma', 0, 20, 1),
        'subsample': hp.quniform('subsample', 0.9, 1.0, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.9, 1.0, 0.01),
        'colsample_bynode': hp.quniform('colsample_bynode', 0.9, 1.0, 0.01),
        'colsample_bylevel': hp.quniform('colsample_bylevel', 0.9, 1.0, 0.01),
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'verbosity': 0,
        'random_state': 1,
        }
    # define hyper parameter keys
    hyper_param_keys = [
        'n_estimators',
        'max_depth',
        'min_child_weight',
        'max_delta_step',
        'learning_rate',
        'reg_lambda',
        'reg_alpha',
        'scale_pos_weight',
        'gamma',
        'subsample',
        'colsample_bytree',
        'colsample_bynode',
        'colsample_bylevel'
        ]
    max_evals = 100
    cutoff = 0.1

    def __init__(self, feat_file, splits, feat_cols, lbl_name, out_dir, mdl_id):
        self.feat_file = feat_file
        self.splits = splits
        self.feat_cols = feat_cols
        self.lbl_name = lbl_name
        self.out_dir = out_dir
        self.mdl_id = mdl_id
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    @classmethod
    def aupr(cls, y_true, y_pred):
        # calculate the Area Under Curve for Precesion- Recall
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        area = metrics.auc(recall, precision)    
        return area

    @classmethod
    def parse_hyperopt_trials(cls, trial_results):
    # turns hyperopt object into a pandas dataframe
        loss = []
        scores = []
        parameters = []

        for t in trial_results.trials:
            try:
                scores.append(t['result']['scores'])
                loss.append(t['result']['loss'])
                parameters.append(t['misc']['vals'])
            except KeyError as e:
                continue

        for d in parameters:   
            for k,i in d.items():
                try:
                    d[k] = i[0]
                except:
                    d[k] = i

        pd_parameters = pd.DataFrame(parameters)
        pd_scores = pd.DataFrame(scores)

        results = pd.concat([pd_parameters,pd_scores], axis = 1)
        results.insert(loc=0, column='loss', value=loss)
        results['parameter_space'] = parameters

        return results

    def train_model(self, params):
        """
        function to create and train the model. This is called by hyper opt.
        It returned the loss (=optimisation metric), status and a dict with 
        supporting information.
        """
        params['n_estimators'] = int(params['n_estimators'])
        clf_xgb = xgb.XGBClassifier()
        clf_xgb = clf_xgb.set_params(**params)
        clf_xgb = clf_xgb.fit(self.X_train, self.y_train)

        pred_train = clf_xgb.predict_proba(self.X_train)[:, 1]
        pred_val = clf_xgb.predict_proba(self.X_val)[:, 1]

        
        aupr_train = XgGridSearch.aupr(self.y_train, pred_train)
        aupr_val = XgGridSearch.aupr(self.y_val, pred_val)
    
        binary_pred_train = [1 if c > XgGridSearch.cutoff else 0 for c in pred_train]
        binary_pred_val = [1 if c > XgGridSearch.cutoff else 0 for c in pred_val]

        f1_train = metrics.f1_score(self.y_train, binary_pred_train)
        f1_val = metrics.f1_score(self.y_val, binary_pred_val)

        xgb_structure = clf_xgb.get_booster().trees_to_dataframe()
        trees_count = len(np.unique(xgb_structure['Tree']))
        splits_sount = xgb_structure['Split'].count()
        nodes_count = xgb_structure.shape[0]

        scores_dict = {
            'aupr_train': aupr_train, 'aupr_val': aupr_val, 
            'trees': trees_count, 'nodes': nodes_count        
        }
        
        return {'loss': -aupr_val, 'status': STATUS_OK, 'scores':scores_dict}

    @classmethod
    def extract_param_dict(cls, param_series):
        # convert a pandas series with the best hyper parameter settings in a dict.
        param_dict = param_series[XgGridSearch.hyper_param_keys].to_dict()

        #fixed keys
        param_dict["random_state"] = 1
        param_dict["objective"] = "binary:logistic" 
        param_dict["tree_method"] = "hist"
        param_dict["booster"] = "gbtree"
        param_dict['n_estimators'] = int(param_dict['n_estimators'])
        return param_dict

    # load data and determine train, validation and test split
    def get_train_test(self, splits):
        feat = pd.read_parquet(self.feat_file)
 
        train = feat.loc[feat['dag'] <= splits[0]]
        val = feat.loc[(feat['dag'] > splits[0]) &
                                (feat['dag'] <= splits[1])]

        print(f'train {train.shape}, val: {val.shape}')
        self.X_train = train[self.feat_cols]
        self.y_train = train[self.lbl_name].astype(int)
        self.X_val = val[self.feat_cols]
        self.y_val = val[self.lbl_name].astype(int)

    def execute_grid_search(self):
        trials = Trials()

        argmin = fmin(
            fn=self.train_model,
            space=XgGridSearch.space,
            algo=tpe.suggest,
            max_evals=XgGridSearch.max_evals,
            trials=trials,
            rstate = np.random.RandomState(1)
        )

        pickle.dump(trials, open(self.out_dir + "trails_" + self.mdl_id, "wb"))
        result_df = XgGridSearch.parse_hyperopt_trials(trials)
        params = XgGridSearch.extract_param_dict(result_df.sort_values(by=["loss"]).reset_index(drop=True).loc[0])    
        return params, result_df

    @classmethod
    def plot_gs_results(cls, res):
        """
        plot AUPR as function of overfitting, model complexity
        """
        res['overfitting'] = res['aupr_train'] - res['aupr_val']
        plt.subplot(121)
        plt.scatter(res['loss'], res['overfitting'])
        plt.title('overfitting')
        plt.subplot(122)
        plt.scatter(res['loss'], res['nodes'])
        plt.title('complexity')
        plt.show()

    def train_final_model(self, params, file_prefix):
        # creates and trains a xgboost model
        xgb_model = xgb.XGBClassifier()
        xgb_model = xgb_model.set_params(**params)
        xgb_model = xgb_model.fit(self.X_train, self.y_train)

        # predict test and train set
        pred_train = xgb_model.predict_proba(self.X_train)[:, 1]
        pred_test = xgb_model.predict_proba(self.X_val)[:, 1]
    
        # store results
        pickle.dump(xgb_model, open(file_prefix+"_mdl.pickle", "wb"))
        pickle.dump(pred_train, open(file_prefix+"_pred_train", "wb"))
        pickle.dump(pred_test, open(file_prefix+"_pred_test", "wb"))
        return xgb_model, pred_train, pred_test

    @classmethod
    def plot_auc(cls, y_train, y_hat_train, y_test, y_hat_test):
        fpr_train, tpr_train, thres_train = roc_curve(y_train, y_hat_train)
        fpr_test, tpr_test, thres_test = roc_curve(y_test, y_hat_test)

        fig, ax = plt.subplots() 
        ax.plot(fpr_train, tpr_train, linestyle='--', label='train')
        ax.plot(fpr_test, tpr_test, linestyle='-.', label='test')

    def controller(self, verbose=True, test_modus=False):
        if test_modus:
            XgGridSearch.max_evals=4
        self.get_train_test(self.splits[0:2])
        params, res_df = self.execute_grid_search()
        if verbose:
            XgGridSearch.plot_gs_results(res_df)
        self.get_train_test(self.splits[1:3])
        xgb_model, pred_train, pred_test = self.train_final_model(params, self.out_dir + self.mdl_id + "_")
        
        auc_test = roc_auc_score(self.y_val, pred_test)
        auc_train = roc_auc_score(self.y_train, pred_train)
        aupr_test = XgGridSearch.aupr(self.y_val, pred_test)
        aupr_train = XgGridSearch.aupr(self.y_train, pred_train)
        print(f'auc train {auc_train}, auc_test {auc_train}, aupr train {aupr_train}, aupr test {aupr_test}')
        if verbose:
            XgGridSearch.plot_auc(self.y_train, pred_train, self.y_val, pred_test)
        gs_res = {'auc_train': auc_train,
               'auc_test': auc_train,
               'aupr_train': aupr_train, 
               'aupr_test': aupr_test}
        return gs_res

# %%


# feat_file = os.getcwd() + "/data/bank_a/features"
# splits = [10 ,17 , 26]
# feat_cols = ['first_half_in', 'second_half_in', 'prior_month_in', 'cnt_in',
#        'first_half_out', 'second_half_out', 'prior_month_out', 'cnt_out', 'embed_0', 'embed_1', 'embed_2',
#        'embed_3', 'embed_4', 'embed_5', 'embed_6', 'embed_7']
# lbl_name = 'is_sar'
# out_dir = os.getcwd() + "/data/bank_a/"
# mdl_id = "test_mld"
# #%%

# gs = XgGridSearch(feat_file, splits, feat_cols, lbl_name, out_dir, mdl_id)
# # %%
# res = gs.controller()
# print(res)
# # %%
