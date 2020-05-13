# coding: utf-8

# 通过分析本次数据（分析过程在之后的文档中说明，发现isNew为区分的两部分数据差异较大，故利用了迁移学习的思想，
# isNew为0的为旧数据，为1的为新数据，在旧数据上第一次训练，新数据和test预测，将结果作为特征，
# 在新数据上进行第二次学习，test上预测，得到最后结果）


import numpy as np
import pandas as pd
import gc, warnings,

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GroupKFold

from tqdm import tqdm

import math

warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("D:/kaggle_dataset/xiamen/train.csv")
test = pd.read_csv("D:/kaggle_dataset/xiamen/test.csv")
train_target = pd.read_csv("D:/kaggle_dataset/xiamen/train_target.csv")
train["target"] = train_target["target"]

train.loc[train.dist > 900000, "dist"] = train.loc[train.dist > 900000, "dist"] - 300000
train.loc[(train.isNew == 0) & (train.residentAddr != -999), "residentAddr"] = train.loc[(train.isNew == 0) & (
            train.residentAddr != -999), "residentAddr"] - 300000

train.loc[train.residentAddr == -999, "residentAddr"] = 200000
test.loc[test.residentAddr == -999, "residentAddr"] = 200000

for data in [train, test]:
    for col in ["certId", "dist", "residentAddr"]:
        data[col] = data[col] - 200000
        data[col + "_pre_1"] = data[col].astype(str).apply(lambda x: x[:1]).astype(int)
        data[col + "_pre_2"] = data[col].astype(str).apply(lambda x: x[:2])
        data[col + "_pre_3"] = data[col].astype(str).apply(lambda x: x[:3])
        data[col + "_pre_4"] = data[col].astype(str).apply(lambda x: x[:4])
        data[col + "_pre_5"] = data[col].astype(str).apply(lambda x: x[:5])
        data[col + "_suffix_3"] = data[col].astype(str).apply(lambda x: x[3:])
        data[col + "_suffix_2"] = data[col].astype(str).apply(lambda x: x[4:])

for data in [train, test]:
    data["dist_certId"] = (data.dist == data.certId).astype(int)
    data["dist_certId_pre_1"] = (data.dist_pre_1 == data.certId_pre_1).astype(int)
    data["dist_certId_pre_2"] = (data.dist_pre_2 == data.certId_pre_2).astype(int)
    data["dist_certId_pre_3"] = (data.dist_pre_3 == data.certId_pre_3).astype(int)
    data["dist_certId_pre_4"] = (data.dist_pre_4 == data.certId_pre_4).astype(int)
    data["dist_certId_pre_5"] = (data.dist_pre_5 == data.certId_pre_5).astype(int)

    data["dist_residentAddr"] = (data.dist == data.residentAddr).astype(int)
    data["dist_residentAddr_pre_1"] = (data.dist_pre_1 == data.residentAddr_pre_1).astype(int)
    data["dist_residentAddr_pre_2"] = (data.dist_pre_2 == data.residentAddr_pre_2).astype(int)
    data["dist_residentAddr_pre_3"] = (data.dist_pre_3 == data.residentAddr_pre_3).astype(int)
    data["dist_residentAddr_pre_4"] = (data.dist_pre_4 == data.residentAddr_pre_4).astype(int)
    data["dist_residentAddr_pre_5"] = (data.dist_pre_5 == data.residentAddr_pre_5).astype(int)

    data["residentAddr_certId"] = (data.residentAddr == data.certId).astype(int)
    data["residentAddr_certId_pre_1"] = (data.residentAddr_pre_1 == data.certId_pre_1).astype(int)
    data["residentAddr_certId_pre_2"] = (data.residentAddr_pre_2 == data.certId_pre_2).astype(int)
    data["residentAddr_certId_pre_3"] = (data.residentAddr_pre_3 == data.certId_pre_3).astype(int)
    data["residentAddr_certId_pre_4"] = (data.residentAddr_pre_4 == data.certId_pre_4).astype(int)
    data["residentAddr_certId_pre_5"] = (data.residentAddr_pre_5 == data.certId_pre_5).astype(int)

for data in [train, test]:
    data["dist_suffix_3_is000"] = (data.dist_suffix_3 == "000").astype(int)
    data["dist_suffix_3_is100"] = (data.dist_suffix_3 == "100").astype(int)
    data["dist_suffix_3_is200"] = (data.dist_suffix_3 == "200").astype(int)
    data["certId_suffix_2_is00"] = (data.certId_suffix_2 == "00").astype(int)
    data["residentAddr_suffix_2_is00"] = (data.residentAddr_suffix_2 == "00").astype(int)

    data["dist_diff_certId"] = np.abs(data["dist"] - data["certId"])

for data in [train, test]:
    data.loc[data.dist_suffix_3.apply(lambda x: x[1:]) != "00", "dist_suffix_3"] = 10
    data.loc[data.dist_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "dist_suffix_3"] = data.loc[
        data.dist_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "dist_suffix_3"].apply(lambda x: x[:1]).astype(
        int)

    data.loc[data.residentAddr_suffix_3.apply(lambda x: x[1:]) != "00", "residentAddr_suffix_3"] = 10
    data.loc[data.residentAddr_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "residentAddr_suffix_3"] = data.loc[
        data.residentAddr_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "residentAddr_suffix_3"].apply(
        lambda x: x[:1]).astype(int)

    data.loc[data.certId_suffix_3.apply(lambda x: x[1:]) != "00", "certId_suffix_3"] = 10
    data.loc[data.certId_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "certId_suffix_3"] = data.loc[
        data.certId_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "certId_suffix_3"].apply(
        lambda x: x[:1]).astype(int)

    data['certBeginDt'] = pd.to_datetime(data["certValidBegin"] * 1000000000) - pd.offsets.DateOffset(years=70)
    data['certStopDt'] = pd.to_datetime(data["certValidStop"] * 1000000000) - pd.offsets.DateOffset(years=70)
    data["certStopDt" + "certBeginDt"] = data["certStopDt"] - data["certBeginDt"]
    data["certStopDtcertBeginDt"] = data["certStopDtcertBeginDt"].apply(lambda x: x.days)

for data in [train, test]:
    data.loc[(data.bankCard.isnull()) | (data.bankCard == -999), "bankCard"] = "999999999."

for data in [train, test]:
    for col in ["bankCard"]:
        data[col + "_pre_1"] = data[col].astype(str).apply(lambda x: x[:1])
        data[col + "_pre_2"] = data[col].astype(str).apply(lambda x: x[:2])
        data[col + "_pre_3"] = data[col].astype(str).apply(lambda x: x[:3])
        data[col + "_pre_4"] = data[col].astype(str).apply(lambda x: x[:4])
        data[col + "_pre_5"] = data[col].astype(str).apply(lambda x: x[:5])
        data[col + "_pre_6"] = data[col].astype(str).apply(lambda x: x[:6])
        data[col + "_suff_3"] = data[col].astype(str).apply(lambda x: x[6:x.find(".")])

i_cols = ["ethnic"]

for col in i_cols:
    temp_df = pd.concat([train[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
    train[col + '_fq_enc'] = train[col].map(fq_encode)
    test[col + '_fq_enc'] = test[col].map(fq_encode)

train["lmt_log"] = np.log(train["lmt"])
test["lmt_log"] = np.log(test["lmt"])

train["product_gender"] = train["loanProduct"] + train["gender"]
train["product_gender_str"] = train["loanProduct"].map(str) + "_" + train["gender"].map(str)

test["product_gender"] = test["loanProduct"] + test["gender"]
test["product_gender_str"] = test["loanProduct"].map(str) + "_" + test["gender"].map(str)

train.loc[train.age > 100, "age"] = -999
test.loc[test.age > 100, "age"] = -999

train.loc[train.job == 16, "job"] = np.nan
test.loc[test.job == 16, "job"] = np.nan

for data in [train, test]:
    for col in ["product_gender_str", "x_33_x_46_str", "x_34_x_52_str"]:
        data[col] = data[col].astype('category')

# 分两次进行迁移学习
train_old = train[train.isNew == 0]
train_new = train[train.isNew == 1]

pd.set_option("max_rows", 100)

for data in [train, test]:
    data["holiday"] = data["weekday"].apply(lambda x: 1 if x == 6 or x == 7 else 0)

# 第一次学习，在旧数据上训练，再新数据和test上预测

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log", "certStopDtcertBeginDt",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"]

features = features_columns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=178)

oof_preds_lgb = np.zeros(train_old[features].shape[0])
y_predss_lgb_train_new = np.zeros(train_new[features].shape[0])
y_predss_lgb_test = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_old[features], train_old["target"])):
    train_x, train_y = train_old[features].iloc[train_idx], train_old["target"].iloc[train_idx]
    valid_x, valid_y = train_old[features].iloc[valid_idx], train_old["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LGBMClassifier(
        boosting_type='gbdt', reg_alpha=0.33, reg_lambda=0.6, missing=-999,
        max_depth=3, n_estimators=1000, objective='binary', metrics='None',
        bagging_fraction=0.8, colsample_bytree=0.8,
        feature_fraction=0.8, learning_rate=0.09, random_state=42,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    y_predss_lgb_train_new += clf.predict_proba(train_new[features], num_iteration=clf.best_iteration_)[:,
                              1] / folds.n_splits
    y_predss_lgb_test += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

print("final auc:", roc_auc_score(train_old["target"], oof_preds_lgb))

# 将预测结果作为特征

train_new["target_rate"] = y_predss_lgb_train_new
test["target_rate"] = y_predss_lgb_test

# 提取第二次学习的新特征

i_cols = ["loanProduct", "bankCard_pre_2", "bankCard_pre_4", "bankCard_pre_6", "dist_suffix_3", "certId_pre_4",
          "bankCard"]

for col in i_cols:
    for agg_type in ["max", "min", "mean", "std"]:
        for amount in ["lmt", "age"]:
            new_col_name = col + '_' + amount + '_' + agg_type
            temp_df = pd.concat([train_new[[col, amount]], test[[col, amount]]])
            temp_df = temp_df.groupby([col])[amount].agg([agg_type]).reset_index().rename(
                columns={agg_type: new_col_name})

            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()

            train_new[new_col_name] = train_new[col].map(temp_df)
            test[new_col_name] = test[col].map(temp_df)

train_new["loanProduct_lmt_diff"] = train_new["lmt"] - train_new["loanProduct_lmt_mean"]
test["loanProduct_lmt_diff"] = test["lmt"] - test["loanProduct_lmt_mean"]

train_new["loanProduct_lmt_rate"] = train_new["lmt"] / train_new["loanProduct_lmt_mean"]
test["loanProduct_lmt_rate"] = test["lmt"] / test["loanProduct_lmt_mean"]

train_new["loanProduct_lmt_rate_max"] = train_new["lmt"] / train_new["loanProduct_lmt_max"]
test["loanProduct_lmt_rate_max"] = test["lmt"] / test["loanProduct_lmt_max"]

train_new["loanProduct_lmt_rate_min"] = train_new["lmt"] / train_new["loanProduct_lmt_min"]
test["loanProduct_lmt_rate_min"] = test["lmt"] / test["loanProduct_lmt_min"]

train_new["lmt_log1p"] = np.log1p(train_new["lmt"])
test["lmt_log1p"] = np.log1p(test["lmt"])

for data in [train_new, test]:
    data["certBeginDt_year"] = data.certBeginDt.apply(lambda x: x.year)
    data["certStopDt_year"] = data.certStopDt.apply(lambda x: x.year)
    data["certBeginDt_month"] = data.certBeginDt.apply(lambda x: x.month)
    data["certStopDt_month"] = data.certStopDt.apply(lambda x: x.month)

    data["certBeginDt_year_2015"] = (data["certBeginDt_year"] == 2015).astype(int)
    data["certBeginDt_year_2016"] = (data["certBeginDt_year"] == 2016).astype(int)
    data["certBeginDt_year_2017"] = (data["certBeginDt_year"] == 2017).astype(int)
    data["certBeginDt_year_2018"] = (data["certBeginDt_year"] == 2018).astype(int)

for data in [train_new, test]：
data["dist_suffix_2_04"] = (data["dist_suffix_2"] == "04").astype(int)
data["dist_suffix_2_30"] = (data["dist_suffix_2"] == "30").astype(int)
data["dist_suffix_2_81"] = (data["dist_suffix_2"] == "81").astype(int)
data["dist_suffix_2_84"] = (data["dist_suffix_2"] == "84").astype(int)

data["certId_suffix_2_04"] = (data["certId_suffix_2"] == "04").astype(int)
data["certId_suffix_2_12"] = (data["certId_suffix_2"] == "12").astype(int)
data["certId_suffix_2_30"] = (data["certId_suffix_2"] == "30").astype(int)
data["certId_suffix_2_85"] = (data["certId_suffix_2"] == "85").astype(int)
data["certId_suffix_2_35"] = (data["certId_suffix_2"] == "35").astype(int)

data["residentAddr_suffix_2_04"] = (data["residentAddr_suffix_2"] == "04").astype(int)
data["residentAddr_suffix_2_06"] = (data["residentAddr_suffix_2"] == "06").astype(int)
data["residentAddr_suffix_2_07"] = (data["residentAddr_suffix_2"] == "07").astype(int)
data["residentAddr_suffix_2_13"] = (data["residentAddr_suffix_2"] == "13").astype(int)
data["residentAddr_suffix_2_14"] = (data["residentAddr_suffix_2"] == "14").astype(int)

i_cols = ["bankCard_pre_6", "bankCard_suff_3"]

for col in i_cols:
    temp_df = pd.concat([train_new[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
    train_new[col + '_fq_enc'] = train_new[col].map(fq_encode)
    test[col + '_fq_enc'] = test[col].map(fq_encode)

# 第二次学习

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log", "certStopDtcertBeginDt",

                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ["target_rate"] + ["dist_suffix_3",
                                                                                              "certId_suffix_3",
                                                                                              "residentAddr_suffix_3", ]
len(features_columns)

features = features_columns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_lgb = np.zeros(train_new[features].shape[0])
y_predss_lgb_test_again = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LGBMClassifier(
        boosting_type='gbdt', reg_alpha=0.33, reg_lambda=0.6, missing=-999,
        max_depth=4, n_estimators=1000, objective='binary', metrics='None',
        bagging_fraction=0.8, colsample_bytree=0.8,
        feature_fraction=0.8, learning_rate=0.09, random_state=42,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    y_predss_lgb_test_again += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:,
                               1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_lgb))

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan", "dist_suffix_2_81",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "certStopDtcertBeginDt", "certBeginDt_year_2016",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ["target_rate"] + ["dist_suffix_3",
                                                                                              "certId_suffix_3",
                                                                                              "residentAddr_suffix_3",
                                                                                              "bankCard_lmt_mean",
                                                                                              "dist_suffix_3_lmt_mean",
                                                                                              "bankCard_lmt_min",
                                                                                              "bankCard_pre_6_lmt_min",
                                                                                              "loanProduct_lmt_rate_min",
                                                                                              "loanProduct_lmt_mean",
                                                                                              ]
len(features_columns)

features = features_columns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_lgb_1 = np.zeros(train_new[features].shape[0])
y_predss_lgb_test_again_1 = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LGBMClassifier(
        boosting_type='gbdt', reg_alpha=0.33, reg_lambda=0.6, missing=-999,
        max_depth=3, n_estimators=1000, objective='binary', metrics='None',
        bagging_fraction=0.8, colsample_bytree=0.8,
        feature_fraction=0.8, learning_rate=0.09, random_state=42,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_lgb_1[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    y_predss_lgb_test_again_1 += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:,
                                 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb_1[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_lgb_1))
oof_preds_lgb_1

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log1p", "certStopDtcertBeginDt",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ["target_rate"] + ["dist_suffix_3",
                                                                                              "certId_suffix_3",
                                                                                              "residentAddr_suffix_3",
                                                                                              ]
len(features_columns)

features = features_columns

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_cat = np.zeros(train_new[features].shape[0])
y_predss_cat_test_again = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = CatBoostClassifier(learning_rate=0.09, random_state=42, depth=3, loss_function='Logloss', n_estimators=500,
                             use_best_model=True,
                             #                              eval_metric = "AUC",
                             )

    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=50)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_cat[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    y_predss_cat_test_again += clf.predict_proba(test[features])[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_cat[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_cat))

temp_df = pd.concat([train_new, test])

one_hot_col = []
for col in ["product_gender_str"]:
    col_one_hot_df = pd.get_dummies(temp_df[col], prefix=col)
    temp_df = pd.concat([temp_df, col_one_hot_df], axis=1)
    one_hot_col.append(col_one_hot_df.columns.tolist())

train_new = temp_df[:train_new.shape[0]]
test = temp_df[train_new.shape[0]:]

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log", "certStopDtcertBeginDt",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ['product_gender_str_1_1',
                                                                            'product_gender_str_1_2',
                                                                            'product_gender_str_2_1',
                                                                            'product_gender_str_2_2',
                                                                            'product_gender_str_3_1',
                                                                            'product_gender_str_3_2'] + [
                       "target_rate"] + ["dist_suffix_3", "certId_suffix_3",
                                         "residentAddr_suffix_3", ]

train_new["certStopDtcertBeginDt"] = train_new["certStopDtcertBeginDt"].fillna(-999)
test["certStopDtcertBeginDt"] = test["certStopDtcertBeginDt"].fillna(-999)

features = features_columns

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from scipy.sparse import hstack

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_xgb = np.zeros(train_new[features].shape[0])
y_predss_xgb = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = XGBClassifier(
        n_estimators=1000,
        boosting_type='gbdt',
        eval_metric="auc",
        eta=0.14,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.33,
        reg_lambda=0.6,
        missing=-999,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    clf.n_estimators = clf.best_iteration

    oof_preds_xgb[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    y_predss_xgb += clf.predict_proba(test[features])[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_xgb[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_xgb))

# 集成学习stacking

train_new_oof = pd.DataFrame({"lgb1": oof_preds_lgb, "lgb2": oof_preds_lgb_1, "cat": oof_preds_cat,
                              "xgb": oof_preds_xgb, "target": train_new.target})
test_pred = pd.DataFrame({"lgb1": y_predss_lgb_test_again, "lgb2": y_predss_lgb_test_again_1,
                          "cat": y_predss_cat_test_again, "xgb": y_predss_xgb})

train_new_oof = pd.DataFrame({"lgb1": oof_preds_lgb, "lgb2": oof_preds_lgb_1, "cat": oof_preds_cat,
                              "xgb": oof_preds_xgb, "target": train_new.target})
test_pred = pd.DataFrame({"lgb1": y_predss_lgb_test_again, "lgb2": y_predss_lgb_test_again_1,
                          "cat": y_predss_cat_test_again, "xgb": y_predss_xgb, })

# 用lr进行stacking

from sklearn.linear_model import LogisticRegression

features = ["lgb1", "lgb2", "cat", "xgb", ]

folds = KFold(n_splits=5, shuffle=True, random_state=82)

oof_preds_all = np.zeros(train_new_oof.shape[0])
y_predss_all = np.zeros(test_pred.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new_oof[features], train_new_oof["target"])):
    train_x, train_y = train_new_oof[features].iloc[train_idx], train_new_oof["target"].iloc[train_idx]
    valid_x, valid_y = train_new_oof[features].iloc[valid_idx], train_new_oof["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    # coding: utf-8

    # 通过分析本次数据（分析过程在之后的文档中说明，发现isNew为区分的两部分数据差异较大，故利用了迁移学习的思想，
    # isNew为0的为旧数据，为1的为新数据，在旧数据上第一次训练，新数据和test预测，将结果作为特征，
    # 在新数据上进行第二次学习，test上预测，得到最后结果）

    import numpy as np
    import pandas as pd
    import gc, warnings,

    from sklearn import metrics
    from sklearn.model_selection import train_test_split, KFold, GroupKFold

    from tqdm import tqdm

    import math

    warnings.filterwarnings('ignore')
    import seaborn as sns
    import matplotlib.pyplot as plt

    train = pd.read_csv("D:/kaggle_dataset/xiamen/train.csv")
    test = pd.read_csv("D:/kaggle_dataset/xiamen/test.csv")
    train_target = pd.read_csv("D:/kaggle_dataset/xiamen/train_target.csv")
    train["target"] = train_target["target"]

    train.loc[train.dist > 900000, "dist"] = train.loc[train.dist > 900000, "dist"] - 300000
    train.loc[(train.isNew == 0) & (train.residentAddr != -999), "residentAddr"] = train.loc[(train.isNew == 0) & (
                train.residentAddr != -999), "residentAddr"] - 300000

    train.loc[train.residentAddr == -999, "residentAddr"] = 200000
    test.loc[test.residentAddr == -999, "residentAddr"] = 200000

    for data in [train, test]:
        for col in ["certId", "dist", "residentAddr"]:
            data[col] = data[col] - 200000
            data[col + "_pre_1"] = data[col].astype(str).apply(lambda x: x[:1]).astype(int)
            data[col + "_pre_2"] = data[col].astype(str).apply(lambda x: x[:2])
            data[col + "_pre_3"] = data[col].astype(str).apply(lambda x: x[:3])
            data[col + "_pre_4"] = data[col].astype(str).apply(lambda x: x[:4])
            data[col + "_pre_5"] = data[col].astype(str).apply(lambda x: x[:5])
            data[col + "_suffix_3"] = data[col].astype(str).apply(lambda x: x[3:])
            data[col + "_suffix_2"] = data[col].astype(str).apply(lambda x: x[4:])

    for data in [train, test]:
        data["dist_certId"] = (data.dist == data.certId).astype(int)
        data["dist_certId_pre_1"] = (data.dist_pre_1 == data.certId_pre_1).astype(int)
        data["dist_certId_pre_2"] = (data.dist_pre_2 == data.certId_pre_2).astype(int)
        data["dist_certId_pre_3"] = (data.dist_pre_3 == data.certId_pre_3).astype(int)
        data["dist_certId_pre_4"] = (data.dist_pre_4 == data.certId_pre_4).astype(int)
        data["dist_certId_pre_5"] = (data.dist_pre_5 == data.certId_pre_5).astype(int)

        data["dist_residentAddr"] = (data.dist == data.residentAddr).astype(int)
        data["dist_residentAddr_pre_1"] = (data.dist_pre_1 == data.residentAddr_pre_1).astype(int)
        data["dist_residentAddr_pre_2"] = (data.dist_pre_2 == data.residentAddr_pre_2).astype(int)
        data["dist_residentAddr_pre_3"] = (data.dist_pre_3 == data.residentAddr_pre_3).astype(int)
        data["dist_residentAddr_pre_4"] = (data.dist_pre_4 == data.residentAddr_pre_4).astype(int)
        data["dist_residentAddr_pre_5"] = (data.dist_pre_5 == data.residentAddr_pre_5).astype(int)

        data["residentAddr_certId"] = (data.residentAddr == data.certId).astype(int)
        data["residentAddr_certId_pre_1"] = (data.residentAddr_pre_1 == data.certId_pre_1).astype(int)
        data["residentAddr_certId_pre_2"] = (data.residentAddr_pre_2 == data.certId_pre_2).astype(int)
        data["residentAddr_certId_pre_3"] = (data.residentAddr_pre_3 == data.certId_pre_3).astype(int)
        data["residentAddr_certId_pre_4"] = (data.residentAddr_pre_4 == data.certId_pre_4).astype(int)
        data["residentAddr_certId_pre_5"] = (data.residentAddr_pre_5 == data.certId_pre_5).astype(int)

    for data in [train, test]:
        data["dist_suffix_3_is000"] = (data.dist_suffix_3 == "000").astype(int)
        data["dist_suffix_3_is100"] = (data.dist_suffix_3 == "100").astype(int)
        data["dist_suffix_3_is200"] = (data.dist_suffix_3 == "200").astype(int)
        data["certId_suffix_2_is00"] = (data.certId_suffix_2 == "00").astype(int)
        data["residentAddr_suffix_2_is00"] = (data.residentAddr_suffix_2 == "00").astype(int)

        data["dist_diff_certId"] = np.abs(data["dist"] - data["certId"])

    for data in [train, test]:
        data.loc[data.dist_suffix_3.apply(lambda x: x[1:]) != "00", "dist_suffix_3"] = 10
        data.loc[data.dist_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "dist_suffix_3"] = data.loc[
            data.dist_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "dist_suffix_3"].apply(
            lambda x: x[:1]).astype(int)

        data.loc[data.residentAddr_suffix_3.apply(lambda x: x[1:]) != "00", "residentAddr_suffix_3"] = 10
        data.loc[data.residentAddr_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "residentAddr_suffix_3"] = \
        data.loc[data.residentAddr_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "residentAddr_suffix_3"].apply(
            lambda x: x[:1]).astype(int)

        data.loc[data.certId_suffix_3.apply(lambda x: x[1:]) != "00", "certId_suffix_3"] = 10
        data.loc[data.certId_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "certId_suffix_3"] = data.loc[
            data.certId_suffix_3.astype(str).apply(lambda x: x[1:]) == "00", "certId_suffix_3"].apply(
            lambda x: x[:1]).astype(int)

        data['certBeginDt'] = pd.to_datetime(data["certValidBegin"] * 1000000000) - pd.offsets.DateOffset(years=70)
        data['certStopDt'] = pd.to_datetime(data["certValidStop"] * 1000000000) - pd.offsets.DateOffset(years=70)
        data["certStopDt" + "certBeginDt"] = data["certStopDt"] - data["certBeginDt"]
        data["certStopDtcertBeginDt"] = data["certStopDtcertBeginDt"].apply(lambda x: x.days)

    for data in [train, test]:
        data.loc[(data.bankCard.isnull()) | (data.bankCard == -999), "bankCard"] = "999999999."

    for data in [train, test]:
        for col in ["bankCard"]:
            data[col + "_pre_1"] = data[col].astype(str).apply(lambda x: x[:1])
            data[col + "_pre_2"] = data[col].astype(str).apply(lambda x: x[:2])
            data[col + "_pre_3"] = data[col].astype(str).apply(lambda x: x[:3])
            data[col + "_pre_4"] = data[col].astype(str).apply(lambda x: x[:4])
            data[col + "_pre_5"] = data[col].astype(str).apply(lambda x: x[:5])
            data[col + "_pre_6"] = data[col].astype(str).apply(lambda x: x[:6])
            data[col + "_suff_3"] = data[col].astype(str).apply(lambda x: x[6:x.find(".")])

    i_cols = ["ethnic"]

    for col in i_cols:
        temp_df = pd.concat([train[[col]], test[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        train[col + '_fq_enc'] = train[col].map(fq_encode)
        test[col + '_fq_enc'] = test[col].map(fq_encode)

    train["lmt_log"] = np.log(train["lmt"])
    test["lmt_log"] = np.log(test["lmt"])

    train["product_gender"] = train["loanProduct"] + train["gender"]
    train["product_gender_str"] = train["loanProduct"].map(str) + "_" + train["gender"].map(str)

    test["product_gender"] = test["loanProduct"] + test["gender"]
    test["product_gender_str"] = test["loanProduct"].map(str) + "_" + test["gender"].map(str)

    train.loc[train.age > 100, "age"] = -999
    test.loc[test.age > 100, "age"] = -999

    train.loc[train.job == 16, "job"] = np.nan
    test.loc[test.job == 16, "job"] = np.nan

    for data in [train, test]:
        for col in ["product_gender_str", "x_33_x_46_str", "x_34_x_52_str"]:
            data[col] = data[col].astype('category')

    # 分两次进行迁移学习
    train_old = train[train.isNew == 0]
    train_new = train[train.isNew == 1]

    pd.set_option("max_rows", 100)

    for data in [train, test]:
        data["holiday"] = data["weekday"].apply(lambda x: 1 if x == 6 or x == 7 else 0)

    # 第一次学习，在旧数据上训练，再新数据和test上预测

    features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                        "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                        "dist_certId_pre_4", "dist_certId_pre_5",
                        "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                        "dist_residentAddr_pre_3",
                        "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                        "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                        "residentAddr_certId_pre_3",
                        "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                        "unpayOtherLoan", "gender", "unpayIndvLoan",
                        "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                        "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                        "setupHour", "lmt_log", "certStopDtcertBeginDt",
                        ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34",
                             "x_41", "x_43",
                             "x_45", "x_46",
                             "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65",
                             "x_66", "x_67",
                             "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"]

    features = features_columns

    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import KFold

    folds = KFold(n_splits=5, shuffle=True, random_state=178)

    oof_preds_lgb = np.zeros(train_old[features].shape[0])
    y_predss_lgb_train_new = np.zeros(train_new[features].shape[0])
    y_predss_lgb_test = np.zeros(test[features].shape[0])
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_old[features], train_old["target"])):
        train_x, train_y = train_old[features].iloc[train_idx], train_old["target"].iloc[train_idx]
        valid_x, valid_y = train_old[features].iloc[valid_idx], train_old["target"].iloc[valid_idx]

        print("训练集坏人数量为：", train_y.sum())
        print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

        print("验证集坏人数量为：", valid_y.sum())
        print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

        clf = LGBMClassifier(
            boosting_type='gbdt', reg_alpha=0.33, reg_lambda=0.6, missing=-999,
            max_depth=3, n_estimators=1000, objective='binary', metrics='None',
            bagging_fraction=0.8, colsample_bytree=0.8,
            feature_fraction=0.8, learning_rate=0.09, random_state=42,
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=50, early_stopping_rounds=100)

        feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
        print(feature_imp)
        oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        y_predss_lgb_train_new += clf.predict_proba(train_new[features], num_iteration=clf.best_iteration_)[:,
                                  1] / folds.n_splits
        y_predss_lgb_test += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

    print("final auc:", roc_auc_score(train_old["target"], oof_preds_lgb))

    # 将预测结果作为特征

    train_new["target_rate"] = y_predss_lgb_train_new
    test["target_rate"] = y_predss_lgb_test

    # 提取第二次学习的新特征

    i_cols = ["loanProduct", "bankCard_pre_2", "bankCard_pre_4", "bankCard_pre_6", "dist_suffix_3", "certId_pre_4",
              "bankCard"]

    for col in i_cols:
        for agg_type in ["max", "min", "mean", "std"]:
            for amount in ["lmt", "age"]:
                new_col_name = col + '_' + amount + '_' + agg_type
                temp_df = pd.concat([train_new[[col, amount]], test[[col, amount]]])
                temp_df = temp_df.groupby([col])[amount].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_new[new_col_name] = train_new[col].map(temp_df)
                test[new_col_name] = test[col].map(temp_df)

    train_new["loanProduct_lmt_diff"] = train_new["lmt"] - train_new["loanProduct_lmt_mean"]
    test["loanProduct_lmt_diff"] = test["lmt"] - test["loanProduct_lmt_mean"]

    train_new["loanProduct_lmt_rate"] = train_new["lmt"] / train_new["loanProduct_lmt_mean"]
    test["loanProduct_lmt_rate"] = test["lmt"] / test["loanProduct_lmt_mean"]

    train_new["loanProduct_lmt_rate_max"] = train_new["lmt"] / train_new["loanProduct_lmt_max"]
    test["loanProduct_lmt_rate_max"] = test["lmt"] / test["loanProduct_lmt_max"]

    train_new["loanProduct_lmt_rate_min"] = train_new["lmt"] / train_new["loanProduct_lmt_min"]
    test["loanProduct_lmt_rate_min"] = test["lmt"] / test["loanProduct_lmt_min"]

    train_new["lmt_log1p"] = np.log1p(train_new["lmt"])
    test["lmt_log1p"] = np.log1p(test["lmt"])

    for data in [train_new, test]:
        data["certBeginDt_year"] = data.certBeginDt.apply(lambda x: x.year)
        data["certStopDt_year"] = data.certStopDt.apply(lambda x: x.year)
        data["certBeginDt_month"] = data.certBeginDt.apply(lambda x: x.month)
        data["certStopDt_month"] = data.certStopDt.apply(lambda x: x.month)

        data["certBeginDt_year_2015"] = (data["certBeginDt_year"] == 2015).astype(int)
        data["certBeginDt_year_2016"] = (data["certBeginDt_year"] == 2016).astype(int)
        data["certBeginDt_year_2017"] = (data["certBeginDt_year"] == 2017).astype(int)
        data["certBeginDt_year_2018"] = (data["certBeginDt_year"] == 2018).astype(int)

    for data in [train_new, test]：
    data["dist_suffix_2_04"] = (data["dist_suffix_2"] == "04").astype(int)
    data["dist_suffix_2_30"] = (data["dist_suffix_2"] == "30").astype(int)
    data["dist_suffix_2_81"] = (data["dist_suffix_2"] == "81").astype(int)
    data["dist_suffix_2_84"] = (data["dist_suffix_2"] == "84").astype(int)

    data["certId_suffix_2_04"] = (data["certId_suffix_2"] == "04").astype(int)
    data["certId_suffix_2_12"] = (data["certId_suffix_2"] == "12").astype(int)
    data["certId_suffix_2_30"] = (data["certId_suffix_2"] == "30").astype(int)
    data["certId_suffix_2_85"] = (data["certId_suffix_2"] == "85").astype(int)
    data["certId_suffix_2_35"] = (data["certId_suffix_2"] == "35").astype(int)

    data["residentAddr_suffix_2_04"] = (data["residentAddr_suffix_2"] == "04").astype(int)
    data["residentAddr_suffix_2_06"] = (data["residentAddr_suffix_2"] == "06").astype(int)
    data["residentAddr_suffix_2_07"] = (data["residentAddr_suffix_2"] == "07").astype(int)
    data["residentAddr_suffix_2_13"] = (data["residentAddr_suffix_2"] == "13").astype(int)
    data["residentAddr_suffix_2_14"] = (data["residentAddr_suffix_2"] == "14").astype(int)

i_cols = ["bankCard_pre_6", "bankCard_suff_3"]

for col in i_cols:
    temp_df = pd.concat([train_new[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
    train_new[col + '_fq_enc'] = train_new[col].map(fq_encode)
    test[col + '_fq_enc'] = test[col].map(fq_encode)

# 第二次学习

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log", "certStopDtcertBeginDt",

                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ["target_rate"] + ["dist_suffix_3",
                                                                                              "certId_suffix_3",
                                                                                              "residentAddr_suffix_3", ]
len(features_columns)

features = features_columns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_lgb = np.zeros(train_new[features].shape[0])
y_predss_lgb_test_again = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LGBMClassifier(
        boosting_type='gbdt', reg_alpha=0.33, reg_lambda=0.6, missing=-999,
        max_depth=4, n_estimators=1000, objective='binary', metrics='None',
        bagging_fraction=0.8, colsample_bytree=0.8,
        feature_fraction=0.8, learning_rate=0.09, random_state=42,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    y_predss_lgb_test_again += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:,
                               1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_lgb))

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan", "dist_suffix_2_81",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "certStopDtcertBeginDt", "certBeginDt_year_2016",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ["target_rate"] + ["dist_suffix_3",
                                                                                              "certId_suffix_3",
                                                                                              "residentAddr_suffix_3",
                                                                                              "bankCard_lmt_mean",
                                                                                              "dist_suffix_3_lmt_mean",
                                                                                              "bankCard_lmt_min",
                                                                                              "bankCard_pre_6_lmt_min",
                                                                                              "loanProduct_lmt_rate_min",
                                                                                              "loanProduct_lmt_mean",
                                                                                              ]
len(features_columns)

features = features_columns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_lgb_1 = np.zeros(train_new[features].shape[0])
y_predss_lgb_test_again_1 = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LGBMClassifier(
        boosting_type='gbdt', reg_alpha=0.33, reg_lambda=0.6, missing=-999,
        max_depth=3, n_estimators=1000, objective='binary', metrics='None',
        bagging_fraction=0.8, colsample_bytree=0.8,
        feature_fraction=0.8, learning_rate=0.09, random_state=42,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_lgb_1[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    y_predss_lgb_test_again_1 += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:,
                                 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb_1[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_lgb_1))
oof_preds_lgb_1

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan", "product_gender_str",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log1p", "certStopDtcertBeginDt",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ["target_rate"] + ["dist_suffix_3",
                                                                                              "certId_suffix_3",
                                                                                              "residentAddr_suffix_3",
                                                                                              ]
len(features_columns)

features = features_columns

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_cat = np.zeros(train_new[features].shape[0])
y_predss_cat_test_again = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = CatBoostClassifier(learning_rate=0.09, random_state=42, depth=3, loss_function='Logloss', n_estimators=500,
                             use_best_model=True,
                             #                              eval_metric = "AUC",
                             )

    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=50)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    oof_preds_cat[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    y_predss_cat_test_again += clf.predict_proba(test[features])[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_cat[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_cat))

temp_df = pd.concat([train_new, test])

one_hot_col = []
for col in ["product_gender_str"]:
    col_one_hot_df = pd.get_dummies(temp_df[col], prefix=col)
    temp_df = pd.concat([temp_df, col_one_hot_df], axis=1)
    one_hot_col.append(col_one_hot_df.columns.tolist())

train_new = temp_df[:train_new.shape[0]]
test = temp_df[train_new.shape[0]:]

features_columns = ["5yearBadloan", "ncloseCreditCard", "unpayNormalLoan",
                    "dist_certId", "dist_certId_pre_1", "dist_certId_pre_2", "dist_certId_pre_3",
                    "dist_certId_pre_4", "dist_certId_pre_5",
                    "dist_residentAddr", "dist_residentAddr_pre_1", "dist_residentAddr_pre_2",
                    "dist_residentAddr_pre_3",
                    "dist_residentAddr_pre_4", "dist_residentAddr_pre_5",
                    "residentAddr_certId", "residentAddr_certId_pre_1", "residentAddr_certId_pre_2",
                    "residentAddr_certId_pre_3",
                    "residentAddr_certId_pre_4", "residentAddr_certId_pre_5",
                    "unpayOtherLoan", "gender", "unpayIndvLoan",
                    "edu", "loanProduct", "ethnic_fq_enc", "product_gender", "highestEdu",
                    "ethnic", "basicLevel", "linkRela", "weekday", "job", "age",
                    "setupHour", "lmt_log", "certStopDtcertBeginDt",
                    ] + ["x_0", "x_12", "x_14", "x_20", "x_25", "x_26", "x_27", "x_28", "x_29", "x_33", "x_34", "x_41",
                         "x_43",
                         "x_45", "x_46",
                         "x_47", "x_48", "x_50", "x_51", "x_52", "x_53", "x_54", "x_61", "x_62", "x_63", "x_65", "x_66",
                         "x_67",
                         "x_68", "x_71", "x_72", "x_74", "x_75", "x_76"] + ['product_gender_str_1_1',
                                                                            'product_gender_str_1_2',
                                                                            'product_gender_str_2_1',
                                                                            'product_gender_str_2_2',
                                                                            'product_gender_str_3_1',
                                                                            'product_gender_str_3_2'] + [
                       "target_rate"] + ["dist_suffix_3", "certId_suffix_3",
                                         "residentAddr_suffix_3", ]

train_new["certStopDtcertBeginDt"] = train_new["certStopDtcertBeginDt"].fillna(-999)
test["certStopDtcertBeginDt"] = test["certStopDtcertBeginDt"].fillna(-999)

features = features_columns

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from scipy.sparse import hstack

folds = KFold(n_splits=5, shuffle=True, random_state=998)

oof_preds_xgb = np.zeros(train_new[features].shape[0])
y_predss_xgb = np.zeros(test[features].shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new[features], train_new["target"])):
    train_x, train_y = train_new[features].iloc[train_idx], train_new["target"].iloc[train_idx]
    valid_x, valid_y = train_new[features].iloc[valid_idx], train_new["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = XGBClassifier(
        n_estimators=1000,
        boosting_type='gbdt',
        eval_metric="auc",
        eta=0.14,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.33,
        reg_lambda=0.6,
        missing=-999,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    clf.n_estimators = clf.best_iteration

    oof_preds_xgb[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    y_predss_xgb += clf.predict_proba(test[features])[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_xgb[valid_idx])))

print("final auc:", roc_auc_score(train_new["target"], oof_preds_xgb))

# 集成学习stacking

train_new_oof = pd.DataFrame({"lgb1": oof_preds_lgb, "lgb2": oof_preds_lgb_1, "cat": oof_preds_cat,
                              "xgb": oof_preds_xgb, "target": train_new.target})
test_pred = pd.DataFrame({"lgb1": y_predss_lgb_test_again, "lgb2": y_predss_lgb_test_again_1,
                          "cat": y_predss_cat_test_again, "xgb": y_predss_xgb})

train_new_oof = pd.DataFrame({"lgb1": oof_preds_lgb, "lgb2": oof_preds_lgb_1, "cat": oof_preds_cat,
                              "xgb": oof_preds_xgb, "target": train_new.target})
test_pred = pd.DataFrame({"lgb1": y_predss_lgb_test_again, "lgb2": y_predss_lgb_test_again_1,
                          "cat": y_predss_cat_test_again, "xgb": y_predss_xgb, })

# 用lr进行stacking

from sklearn.linear_model import LogisticRegression

features = ["lgb1", "lgb2", "cat", "xgb", ]

folds = KFold(n_splits=5, shuffle=True, random_state=82)

oof_preds_all = np.zeros(train_new_oof.shape[0])
y_predss_all = np.zeros(test_pred.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_new_oof[features], train_new_oof["target"])):
    train_x, train_y = train_new_oof[features].iloc[train_idx], train_new_oof["target"].iloc[train_idx]
    valid_x, valid_y = train_new_oof[features].iloc[valid_idx], train_new_oof["target"].iloc[valid_idx]

    print("训练集坏人数量为：", train_y.sum())
    print("训练集坏人比例为：", (train_y.sum() / train_x.shape[0]))

    print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LogisticRegression()

    clf.fit(train_x, train_y)

    feature_imp = pd.DataFrame(sorted(zip(clf.coef_[0], train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)

    oof_preds_all[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    y_predss_all += clf.predict_proba(test_pred[features])[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_all[valid_idx])))

print("final auc:", roc_auc_score(train_new_oof["target"], oof_preds_all))

test["target"] = y_predss_all
test[["id", "target"]].to_csv("D:/kaggle_dataset/xiamen/res_stacking.csv", index=False)

print("验证集坏人数量为：", valid_y.sum())
    print("验证集坏人比例为：", (valid_y.sum() / valid_x.shape[0]))

    clf = LogisticRegression()

    clf.fit(train_x, train_y)

    feature_imp = pd.DataFrame(sorted(zip(clf.coef_[0], train_x.columns)), columns=['Value', 'Feature'])
    print(feature_imp)

    oof_preds_all[valid_idx] = clf.predict_proba(valid_x)[:, 1]
    y_predss_all += clf.predict_proba(test_pred[features])[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_all[valid_idx])))

print("final auc:", roc_auc_score(train_new_oof["target"], oof_preds_all))

test["target"] = y_predss_all
test[["id", "target"]].to_csv("D:/kaggle_dataset/xiamen/res_stacking.csv", index=False)

