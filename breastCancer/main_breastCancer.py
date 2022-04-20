# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:22:45 2022

@author: Alex
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

df=pd.read_csv("breast-cancer-diagnostic.shuf.lrn.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['class']])

print("------- KNN ---------")


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train.drop("class",axis=1), train["class"])

proba=neigh.predict_proba(test.drop("class",axis=1))[:, 1]
pred=neigh.predict(test.drop("class",axis=1))

print(roc_auc_score(test["class"], proba))
print("rechts prediction")
print(confusion_matrix(test["class"], pred))
print(cross_val_score(neigh, df.drop("class",axis=1), df["class"], scoring="roc_auc", cv = 5))
print(sum(cross_val_score(neigh, df.drop("class",axis=1), df["class"], scoring="roc_auc", cv = 5))/5)

print("------- LDA ---------")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
#lda.fit(X, y)
lda.fit(train.drop("class",axis=1), train["class"])

proba=lda.predict_proba(test.drop("class",axis=1))[:, 1]
pred=lda.predict(test.drop("class",axis=1))

print(roc_auc_score(test["class"], proba))
print("rechts prediction")
print(confusion_matrix(test["class"], pred))
print(cross_val_score(lda, df.drop("class",axis=1), df["class"], scoring="roc_auc", cv = 5))
print(sum(cross_val_score(lda, df.drop("class",axis=1), df["class"], scoring="roc_auc", cv = 5))/5)


print("------- RFC ---------")

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2, random_state=0)

rfc.fit(train.drop("class",axis=1), train["class"])

proba=rfc.predict_proba(test.drop("class",axis=1))[:, 1]
pred=rfc.predict(test.drop("class",axis=1))

print(roc_auc_score(test["class"], proba))
print("rechts prediction")
print(confusion_matrix(test["class"], pred))
print(cross_val_score(rfc, df.drop("class",axis=1), df["class"], scoring="roc_auc", cv = 5))
print(sum(cross_val_score(rfc, df.drop("class",axis=1), df["class"], scoring="roc_auc", cv = 5))/5)






#roc_auc_score(y, clf.predict_proba(X)[:, 1])

#print(cross_val_score(dtree, X, y, scoring="roc_auc", cv = 7))


#confusion_matrix(y_true, y_pred)
#.ravel #(tn, fp, fn, tp)
#




'''def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['mse', 'mae'])
    bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
    max_depth = trial.suggest_int('max_depth', 1, 10000)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 10000)
    n_estimators =  trial.suggest_int('n_estimators', 30, 1000)
    
    regr = RandomForestRegressor(bootstrap = bootstrap, criterion = criterion,
                                 max_depth = max_depth, max_features = max_features,
                                 max_leaf_nodes = max_leaf_nodes,n_estimators = n_estimators,n_jobs=2)
    
    
    #regr.fit(X_train, y_train)
    #y_pred = regr.predict(X_val)
    #return r2_score(y_val, y_pred)
    
    score = cross_val_score(regr, X_train, y_train, cv=5, scoring="r2")
    r2_mean = score.mean()

    return r2_mean
#Execute optuna and set hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

#Create an instance with tuned hyperparameters
optimised_rf = RandomForestRegressor(bootstrap = study.best_params['bootstrap'], criterion = study.best_params['criterion'],
                                     max_depth = study.best_params['max_depth'], max_features = study.best_params['max_features'],
                                     max_leaf_nodes = study.best_params['max_leaf_nodes'],n_estimators = study.best_params['n_estimators'],
                                     n_jobs=2)
#learn
optimised_rf.fit(X_train ,y_train)'''



