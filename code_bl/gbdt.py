# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-04-17 19:34:38 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-04-17 19:34:38 
"""
import numpy as np

import lightgbm as lgb

import gen_features

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from time import gmtime, strftime


def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def submit_result(submit, result, model_name):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(
        '../submit/{}_result_{}.csv'.format(model_name, now_time), index=False)


def train_lgb(train_x, train_y, test_x):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
    }
    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']
    scores = []
    result_proba = []
    for tr_idx, val_idx in kfold.split(train_x, train_y):
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=cate_cols)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=cate_cols)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=50, num_boost_round=40000, verbose_eval=50, feval=eval_f)
        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        result_proba.append(lgb_model.predict(
            test_x, num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    return pred_test


def hyperparameter_seach(train_x, train_y):
    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform

    fit_params={"early_stopping_rounds":30, 
                "eval_metric" : 'multiclass', 
                "eval_set" : [(train_x, train_y)],
                'eval_names': ['valid'],
                'verbose': 100,
                'categorical_feature': ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                                        'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                                        'first_mode', 'weekday', 'hour']}

    param_test ={'num_leaves': sp_randint(6, 50), 
                 'min_child_samples': sp_randint(100, 500), 
                 'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                 'subsample': sp_uniform(loc=0.2, scale=0.8), 
                 'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                 'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                 'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    #This parameter defines the number of HP points to be tested
    n_HP_points_to_test = 100

    import lightgbm as lgb
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

    # n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the           # absolute maximum
    clf = lgb.LGBMClassifier(max_depth=-1,
                             random_state=314,
                             silent=True,
                             n_jobs=4,
                             n_estimators=5000)
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test, 
        n_iter=n_HP_points_to_test,
        scoring='f1',
        cv=3,
        refit=True,
        random_state=314,
        verbose=True)

    gs.fit(train_x, train_y, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


if __name__ == '__main__':
    data, train_x, train_y, test_x, submit = gen_features.get_train_test_feas_data()
    hyperparameter_seach(train_x, train_y)
    #result_lgb = train_lgb(train_x, train_y, test_x)
    #submit_result(submit, result_lgb, 'lgb')
