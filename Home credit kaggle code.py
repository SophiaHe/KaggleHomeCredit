# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:04:59 2018

@author: hyq92
"""

import pandas as pd
import numpy as np
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# import lifelines

application_train = pd.read_csv("application_train.csv", header=0)
application_test = pd.read_csv("application_test.csv", header=0)
list(application_train)

application_train.ix[1:5, 0:2]

##################### EDA #####################
# out of 307511, 282686 repaid & 24825 had payment difficulties 
application_train["TARGET"].value_counts() 






############## Label Encoding and One-Hot Encoding ##############
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in application_train:
    if application_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(application_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(application_train[col])
            # Transform both training and testing data
            application_train[col] = le.transform(application_train[col])
            application_test[col] = le.transform(application_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)

print('Training Features shape: ', application_train.shape)
print('Testing Features shape: ', application_test.shape)

################ Aligning Training and Testing Data ################
# One-hot encoding has created more columns in the training data because there were 
# some categorical variables with categories not represented in the testing data. 
# To remove the columns in the training data that are not in the testing data, 
# we need to align the dataframes. First we extract the target column from the training 
# data (because this is not in the testing data but we need to keep this information). 
# When we do the align, we must make sure to set axis = 1 to align the dataframes based 
# on the columns and not on the rows!

train_labels = application_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
application_train, application_test = application_train.align(application_test, join = 'inner', axis = 1)

print('Training Features shape: ', application_train.shape)
print('Testing Features shape: ', application_test.shape)


############ Back to EDA ############
# Age information into a separate dataframe
age_data = application_train[["TARGET", "DAYS_BIRTH"]]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)



################ Analysis ###############
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import gc

# Format the training and testing data 
train = np.array(application_train)
# train = np.array(application_train.drop('TARGET', axis=1))
test = np.array(application_test)

train_labels = application_train['TARGET']
train_labels = np.array(train_labels).reshape((-1, ))


# 10 fold cross validation
folds = KFold(n_splits=5, shuffle=True, random_state=50)

# Validation and test predictions
valid_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])

# Iterate through each fold
for n_fold, (train_indices, valid_indices) in enumerate(folds.split(train)):
    # Training data for the fold
    train_fold, train_fold_labels = train[train_indices, :], train_labels[train_indices]
    
    # Validation data for the fold
    valid_fold, valid_fold_labels = train[valid_indices, :], train_labels[valid_indices]
    
    # LightGBM classifier with hyperparameters
    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.1,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2
    )
    
    # Fit on the training data, evaluate on the validation data
    clf.fit(train_fold, train_fold_labels, 
            eval_set= [(train_fold, train_fold_labels), (valid_fold, valid_fold_labels)], 
            eval_metric='auc', early_stopping_rounds=100, verbose = False
           )
    
    # Validation preditions
    valid_preds[valid_indices] = clf.predict_proba(valid_fold, num_iteration=clf.best_iteration_)[:, 1]
    
    # Testing predictions
    test_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    # Display the performance for the current fold
    print('Fold %d AUC : %0.6f' % (n_fold + 1, roc_auc_score(valid_fold_labels, valid_preds[valid_indices])))
    
    # Delete variables to free up memory
    del clf, train_fold, train_fold_labels, valid_fold, valid_fold_labels
    gc.collect()
    

# Make a submission dataframe
submission = application_test[['SK_ID_CURR']]
submission['TARGET'] = test_preds

# Save the submission file
submission.to_csv("light_gbm_baseline.csv", index=False)