from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,  LinearSVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, precision_score
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.datasets import make_classification
import xgboost as xgb
import numpy as np
import pandas as pd
rus = RandomUnderSampler(random_state=2021)

def calculate_precision(predict, real):
    cnt = 0
    print(type(predict), type(real))
    real = list(real)
    for i in range(len(predict)):
        if predict[i] == 1 and real[i] == 1:
            cnt += 1
    print(cnt, np.sum(predict))
    return cnt, np.sum(predict), cnt / np.sum(predict)

# SVM
def svm_classification(train, target, svm, k=5):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2021)
    y_predicts = []
    ys = []
    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train.iloc[train_index, :], train.iloc[test_index, :]
        svm.fit(train_X, train_y)
        y_predict = svm.predict(test_X)
        y_predicts.append(y_predict)
        ys.append(test_y)
        print('epoch {} precision_score {}'.format(i, precision_score(test_y, y_predict, average='macro')))
    return y_predicts, ys
    
def svm_classification_2(X, svm, feat, k=5):
    y_predicts = []
    ys = []
    folds = KFold(n_splits=k, shuffle=True, random_state=2021)
    no = X['no'].drop_duplicates()
    for i, (idx_1, idx_2) in enumerate(folds.split(no)):
        trn_data = X[X['no'].isin(no.iloc[idx_1])]
        tst_data = X[X['no'].isin(no.iloc[idx_2])]
        train_X, train_y = trn_data[feat], trn_data['label']
        test_X, test_y = tst_data[feat], tst_data['label']
        svm.fit(train_X, train_y)
        y_predict = svm.predict(test_X)
        y_predicts.append(y_predict)
        ys.append(test_y)
        print('epoch {} precision_score {}'.format(i, precision_score(test_y, y_predict, average='macro')))
    return y_predicts, ys

# LGB
def precision(y_true, y_predict):
    a = list(y_true > 0.5)
    b = list(y_predict.get_label())
    # print(a, b)
    score = precision_score(a, b)
    return 'precision', score, True


def lgb_classification(train, target, test, k=5):
    oof_preds = np.zeros((train.shape[0], ))
    oof_probs = np.zeros((train.shape[0], ))
    feature_importance_df = pd.DataFrame()
    offline_score = []
    output_preds = []
    aa = []
    xx = []
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2021)
    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train.iloc[train_index, :], train.iloc[test_index, :]
        dtrain = lgb.Dataset(train_X, label=train_y)
        dval = lgb.Dataset(test_X, label=test_y)

        parameters = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            # 'num_class': 2,
            'metric': 'None',
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'nthread': 12,
            'random_state': 2021,

        }
        # 'metric': {'binary_logloss', 'auc'}
        lgb_model = lgb.train(
            parameters,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=precision,
        )
        a = lgb_model.predict(
            test_X, num_iteration=lgb_model.best_iteration) > 0.5
        aa.append(a)
        xx.append(test_y)
        oof_preds[test_index] = lgb_model.predict(
            test_X, num_iteration=lgb_model.best_iteration) > 0.6
        output_preds.append(lgb_model.predict(
            test, num_iteration=lgb_model.best_iteration))
        offline_score.append(lgb_model.best_score['valid'])

        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train.columns
        fold_importance_df["importance"] = lgb_model.feature_importance(
            importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
    print('feature importance:')
    a = feature_importance_df.groupby(
        ['feature'])['importance'].mean().sort_values(ascending=False)
    print(a.head(15))
    print('confusion matrix:')
    b = confusion_matrix(target, oof_preds)
    print(b)
    print('classfication report:')
    c = classification_report(target, oof_preds)
    print(c)
    # return oof_probs, np.mean(offline_score)
    return a, b, c, output_preds, aa, xx


def lgb_classification_2(X, test, feat, k=5):
    feature_importance_df = pd.DataFrame()
    offline_score = []
    output_preds = []
    aa = []
    xx = []
    folds = KFold(n_splits=k, shuffle=True, random_state=2021)
    no = X['no'].drop_duplicates()
    for i, (idx_1, idx_2) in enumerate(folds.split(no)):
        trn_data = X[X['no'].isin(no.iloc[idx_1])]
        tst_data = X[X['no'].isin(no.iloc[idx_2])]
        train_X, train_y = trn_data[feat], trn_data['label']
        test_X, test_y = tst_data[feat], tst_data['label']
        dtrain = lgb.Dataset(train_X, label=train_y)
        dval = lgb.Dataset(test_X, label=test_y)

        parameters = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            # 'num_class': 2,
            'metric': 'None',
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'nthread': 12,
            'random_state': 2021,

        }
        # 'metric': {'binary_logloss', 'auc'}
        lgb_model = lgb.train(
            parameters,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=precision,
        )
        a = lgb_model.predict(
            test_X, num_iteration=lgb_model.best_iteration) > 0.5
        aa.append(a)
        xx.append(test_y)
        output_preds.append(lgb_model.predict(
            test, num_iteration=lgb_model.best_iteration))
        offline_score.append(lgb_model.best_score['valid'])

        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feat
        fold_importance_df["importance"] = lgb_model.feature_importance(
            importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
    print('feature importance:')
    a = feature_importance_df.groupby(
        ['feature'])['importance'].mean().sort_values(ascending=False)
    print(a.head(15))
    # return oof_probs, np.mean(offline_score)
    return a, output_preds, aa, xx


# XGB
def xgb_precision(y_true, y_predict):
    a = list(y_true > 0.5)
    b = list(y_predict.get_label())
    score = precision_score(a, b)
    return 'precision', score


def xgb_classification(X, test, feat, k=5):
    # feature_importance_df = pd.DataFrame()
    # offline_score = []
    output_preds = []
    # aa = []
    # xx = []
    folds = KFold(n_splits=k, shuffle=True, random_state=2021)
    no = X['no'].drop_duplicates()
    for i, (idx_1, idx_2) in enumerate(folds.split(no)):
        trn_data = X[X['no'].isin(no.iloc[idx_1])]
        tst_data = X[X['no'].isin(no.iloc[idx_2])]
        train_X, train_y = trn_data[feat], trn_data['label']
        test_X, test_y = tst_data[feat], tst_data['label']

        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'gamma': 0.1,
            'max_depth': 8,
            # 'alpha': 0,
            # 'lambda': 0,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'min_child_weight': 3,
            'silent': 0,
            'eta': 0.01,
            'nthread': -1,
            'seed': 2021,
        }

        dtrain = xgb.DMatrix(train_X, label=train_y)
        dtest = xgb.DMatrix(test_X, label=test_y)
        evals = [(dtrain, 'train'), (dtest, 'valid')]

        xgb_model = xgb.train(params, dtrain, num_boost_round=2000,
                              evals=evals, early_stopping_rounds=100, 
                              verbose_eval=100, feval=xgb_precision)
        oof_pred = xgb_model.predict(xgb.DMatrix(test), ntree_limit = xgb_model.best_ntree_limit)
        output_preds.append(oof_pred)
        y_predict = xgb_model.predict(xgb.DMatrix(test_X), ntree_limit = xgb_model.best_ntree_limit) > 0.5
        print('epoch {} precision : {}'.format(i, precision_score(test_y, y_predict)))
    return output_preds
