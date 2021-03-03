import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, ClassifierMixin, MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt

def get_data(path, mode='train'):

    f = open(path, "r")
    col = []
    label_max = 0
    for i in tqdm(f):
        arr = i.split("|,|")
        if mode=="train":
            arr[-1] = arr[-1].split("\n")[0]
            if len(arr[-1])>0:
                arr[-1] = arr[-1][:-1]
            arr[-2] = arr[-2][:-1]
        else:
            arr[-1] = arr[-1][:-1]
        col.append(arr)

    f.close()
    
    return col


def gen_label(train):
    max_label = train[train['label'].map(len)>0]['label'].map(lambda x:np.max([int(i) for i in x.split(" ")])).max()
    train['label'] = train['label'].replace("", str(max_label+1))
    
    train['label'] = train['label'].map(lambda x:[int(i) for i in x.split(" ")])
    col = np.zeros((train.shape[0], max_label+2))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1
        
    return col

train = pd.DataFrame(get_data("../input/train.csv", "train"), columns=['report_ID', 'description', 'label'])
label = gen_label(train)
target = pd.DataFrame(label, columns=["label_{}".format(i) for i in range(label.shape[1])])
test = pd.DataFrame(get_data("../input/testA.csv", "test"), columns=['report_ID', 'description',])
df = pd.concat([train, test], axis=0, ignore_index=True)
print(train.shape, test.shape)

tfidf = TfidfVectorizer(ngram_range=(1, 5))
tfidf_feature = tfidf.fit_transform(df['description'])
svd_feature = TruncatedSVD(n_components=100).fit_transform(tfidf_feature)
train_df = svd_feature[:-len(test)]
test_df = svd_feature[-len(test):]

scores = []

nfold = 5
kf = MultilabelStratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)

lr_oof = np.zeros(label.shape)
lr_predictions = np.zeros((len(test), label.shape[1]))

i = 0
for train_index, valid_index in kf.split(train_df, label):
    print("\nFold {}".format(i + 1))
    X_train, label_train = train_df[train_index], label[train_index]
    X_valid, label_valid = train_df[valid_index], label[valid_index]

    base = LogisticRegression(C=1)
#     base = SGDClassifier(loss='log')
#     base = SVC(probability = 1)
    model = OneVsRestClassifier(base, n_jobs=20)
#     model = ClassifierChain(base)
    model.fit(X_train, label_train)

    lr_oof[valid_index] = model.predict_proba(X_valid,)
    scores.append(roc_auc_score(label_valid[:,:-1,], lr_oof[valid_index][:,:-1,]))
    
    lr_predictions += model.predict_proba(test_df) / nfold
    i += 1
    print(scores)
    
print(np.mean(scores))

submit = test[['report_ID']]
submit['Prediction'] = [" ".join([str(j) for j in i]) for i in lr_predictions[:, :-1]]
f = open("./setting/LR + TFIDF + CountVector + HashVector + SVD(200).csv", "w")
for index, rows in submit.iterrows():
    f.write("{}|,|{}\n".format(rows['report_ID'], rows['Prediction']))
f.close()