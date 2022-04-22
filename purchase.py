# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:34:20 2022

@author: Alex
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.externals import joblib

#d_train = pd.read_csv("H:/Markus/purchase_ds/purchase600-100cls-15k.lrn.csv")

#d_test = pd.read_csv("H:/Markus/purchase_ds/purchase600-100cls-15k.tes.csv")

#d_sol_ex = pd.read_csv("H:/Markus/purchase_ds/purchase600-100cls-15k.sol.ex.csv")


d_train = pd.read_csv("./data/purchase600-100cls-15k.lrn.csv")

d_test = pd.read_csv("./data/purchase600-100cls-15k.tes.csv")

d_sol_ex = pd.read_csv("./data/purchase600-100cls-15k.sol.ex.csv")

X = d_train.iloc[:,1:601].values
y = d_train.iloc[:,601].values
X_kaggle_test = d_test.iloc[:,1:601].values


#Check if there are any duplicates in the dataset
boolean = d_train.duplicated(subset=['ID']).any()

if boolean == True:
    print("There are duplicates in column ID")
else:
    print("No duplicates found in column ID")
    
   
    
#d_train["class"].describe()
#Calculate and Plot number of Customers per Class
count_per_classes = d_train.groupby(['class'])['ID'].count().sort_values()
count_per_classes.plot(kind="bar", title="Distribution of Customer Classes", figsize=(20,4), ylabel="Count")

#Summarize the Number of each product in the dataset
sum_of_bought_prod = d_train.drop(columns=['ID', 'class']).sum(axis=0)
print(sum_of_bought_prod.sort_values())#.plot(kind="bar", title="Distribution of Bought Products", color="r")
sum_of_bought_prod.describe()
count_per_classes.describe()


###### Random Forest #############

from sklearn.ensemble import RandomForestClassifier
# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 21)

###########  Random Forest Classification ##############
'''rf_classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 42)
rf_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = rf_classifier.predict(X_test)



#print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

test_kaggle_pred = rf_classifier.predict(X_kaggle_test)

df = pd.DataFrame(test_kaggle_pred, d_test["ID"])
df.columns = ["class"]
df.to_csv('./data/output_df.csv')'''



'''# Feature Importance
feature_names = d_train.drop(["ID","class"], axis=1).columns.tolist()
importances = rf_classifier.feature_importances_
## Put in a pandas dtf
dtf_importances = pd.DataFrame({'IMPORTANCE':importances, 
            'VARIABLE':feature_names}).sort_values('IMPORTANCE', 
            ascending=False)
dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index('VARIABLE')
print(dtf_importances.head())

## Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')
dtf_importances[['IMPORTANCE']].sort_values(by='IMPORTANCE').plot(figsize=(20,10), kind='barh', legend=False, ax=ax[0]).grid(axis='x')
ax[0].set(ylabel='')
ax[1].title.set_text('cumulative')
dtf_importances[['cumsum']].plot(kind='line', linewidth=4, 
                                 legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), 
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()

# Reducing Number of features

print(len(dtf_importances))
variable_subset = dtf_importances[dtf_importances["IMPORTANCE"]>0.002]
print(len(variable_subset))

X_names = variable_subset.index.tolist()
X_reduced = d_train[X_names].values

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size = 0.20, random_state = 21)

# Random Forest Classification
rf_classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 42)
rf_classifier.fit(X_train_reduced, y_train_reduced)



# Predicting the Test set results
y_pred_reduced = rf_classifier.predict(X_test_reduced)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(y_test_reduced,y_pred_reduced))
print(classification_report(y_test_reduced,y_pred_reduced))
print(accuracy_score(y_test_reduced, y_pred_reduced))'''



###### lightgbm #############
# Try Gradient Boosting (lightgbm)

import lightgbm as lgb



#clf = lgb.LGBMClassifier().fit(X_train, y_train)

#y_pred = clf.predict(X_test)

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print(accuracy_score(y_test, y_pred))

###### LDA #############


'''from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()
lda_object = lda.fit(X_train, y_train)
X_lda = lda_object.transform(X_train)

y_pred = lda.predict(X_test)



print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

###### QDA #############


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

qda = QDA()
qda = qda.fit(X_train, y_train)


y_pred = qda.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


###### SVM #############

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''


###### Logistic Regression #############


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


###### Logistic Regression with pca #############


from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(n_components=11)


pca.fit(X_train)

train_pca = pca.transform(X_train)
test_pca = pca.transform(X_test)

print(pca.explained_variance_ratio_)


clf = LogisticRegression(random_state=0)
clf.fit(train_pca, y_train)


y_pred = clf.predict(test_pca)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

###### Ridgle Classifier #############


# SUXX


'''from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''

###### Ridgle Classifier #############


# SUXX


'''from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''


###### TPot #######


# check tpot version
'''import tpot
from sklearn.model_selection import RepeatedStratifiedKFold

# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
model = tpot.TPOTClassifier(generations=3, population_size=20, cv=cv, scoring='accuracy', 
                            verbosity=2, random_state=42, n_jobs=-1)

model.fit(X, y)'''


#### Submission 

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(n_components=11)


pca.fit(X)

train_pca = pca.transform(X)

print(pca.explained_variance_ratio_)


clf = LogisticRegression(random_state=0)
clf.fit(train_pca, y)

test_pca = pca.transform(X_kaggle_test)
y_pred = clf.predict(test_pca)



test_kaggle_pred = clf.predict(test_pca)

df = pd.DataFrame(test_kaggle_pred, d_test["ID"])
df.columns = ["class"]
df.to_csv('./data/output_df.csv')





