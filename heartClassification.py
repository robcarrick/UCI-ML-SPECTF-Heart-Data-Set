# =============================================================================
# Binary Classification: Testing a few ML models
# =============================================================================
# Import libraries
import numpy as np
import itertools
import pandas as pd
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold

# =============================================================================
# Import data
# =============================================================================
seed = 1
train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train",
                    header=None)
train = train.sample(frac = 1, random_state = seed) # This shuffles the data
test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test",
                    header=None)
x_train = train.drop(0,axis=1); y_train = train.iloc[:,0]
x_test = test.drop(0,axis=1); y_test = test.iloc[:,0]

# =============================================================================
# Adaboost 
# =============================================================================
# set up the grid of hyper parameters
depth = np.array(np.arange(1,10))
est = np.array(np.arange(1,16))
x = itertools.product(depth, est)
paramGrid = np.array(np.zeros((len(depth)*len(est),3))-1)
i = 0
for j in x:
    paramGrid[i,0:2] = j
    i = i + 1

# evaluate the model for all hyper param combos
for i in range(len(paramGrid)):
    max_depth = int(paramGrid[i,0])
    n_estimators = int(paramGrid[i,1])
    kfold = KFold(n_splits=5, random_state=seed)
    modAdaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                     n_estimators=n_estimators)
    scores = cross_val_score(modAdaboost, x_train, y_train, cv=kfold)
    paramGrid[i,2] = np.mean(scores)
    print(i)

bestParams = paramGrid[:,0:2][np.where(paramGrid[:,2] == max(paramGrid[:,2]))]
bestParams = bestParams.flatten().astype(int)

# Evaluate on the test set with the best hyper params
modAdaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=bestParams[0]),
                                     n_estimators=bestParams[1])
modAdaboost.fit(x_train,y_train) # train model on training set
preds = modAdaboost.predict(x_test) # now predict on the test set
testAccAdaboost = np.mean(y_test == preds)

# =============================================================================
# SVM
# =============================================================================
# set up the grid of hyper parameters
kernels = np.array(['linear','poly','rbf','sigmoid'])
paramGrid = np.array(np.zeros(len(kernels))-1)

i = 0
for kernel in kernels:
    kfold = KFold(n_splits=5, random_state=seed)
    modSVM = svm.SVC(gamma = 'scale', kernel = kernel)
    scores = cross_val_score(modSVM, x_train, y_train, cv=kfold)
    paramGrid[i] = np.mean(scores)
    i += 1
    print(i)

bestParams = kernels[np.where(paramGrid == max(paramGrid))]

# Evaluate on the test set with the best hyper params
modSVM = svm.SVC(gamma='scale', kernel = bestParams) # use default params
modSVM.fit(x_train,y_train)
preds = modSVM.predict(x_test) # now predict on the test set
testAccSVM = np.mean(y_test == preds)

# =============================================================================
# Random Forest
# =============================================================================
# set up the grid of hyper parameters
max_depth = np.array(np.arange(1,9))
max_features = np.array([0.1,0.2,
                         np.sqrt(x_train.shape[1])/x_train.shape[1],
                         0.3,0.4,0.5])                
x = itertools.product(max_depth, max_features)
paramGrid = np.array(np.zeros((len(max_depth)*len(max_features),3))-1)
i = 0
for j in x:
    paramGrid[i,0:2] = j
    i = i + 1

# evaluate the model for all hyper param combos
for i in range(len(paramGrid)):
    max_depth = int(paramGrid[i,0])
    max_features = paramGrid[i,1]
    kfold = KFold(n_splits=5, random_state=seed)
    modRF = RandomForestClassifier(n_estimators=500,
                                   max_depth = max_depth,
                                   max_features = max_features)
    scores = cross_val_score(modRF, x_train, y_train, cv=kfold)
    paramGrid[i,2] = np.mean(scores)
    print(i)

bestParams = paramGrid[:,0:2][np.where(paramGrid[:,2] == max(paramGrid[:,2]))]
bestParams = bestParams.flatten()

# Evaluate on the test set with the best hyper params
modRF = RandomForestClassifier(n_estimators=500,
                               max_depth = int(bestParams[0]),
                               max_features = bestParams[1])
modRF.fit(x_train,y_train) # train model on training set
preds = modRF.predict(x_test) # now predict on the test set
testAccRF = np.mean(y_test == preds)

# =============================================================================
# Logistic Regression
# =============================================================================
modGLM = LogisticRegression()
modGLM.fit(x_train, y_train)
preds = modGLM.predict(x_test) # now predict on the test set
testAccGLM = np.mean(y_test == preds)

# =============================================================================
# KNN
# =============================================================================
# set up the grid of hyper parameters
k = np.array(np.arange(1,51))
paramGrid = np.array(np.zeros((len(k),2))-1)
paramGrid[:,0] = k

# evaluate the model for all hyper param combos
for i in range(len(paramGrid)):
    k = int(paramGrid[i,0])
    kfold = KFold(n_splits=5, random_state=seed)
    modKNN = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(modKNN, x_train, y_train, cv=kfold)
    paramGrid[i,1] = np.mean(scores)
    print(i)

bestParams = paramGrid[:,0][np.where(paramGrid[:,1] == max(paramGrid[:,1]))]
bestParams = bestParams.flatten()[-1] # if there is a tie, pick the largest k
                                      # is a simpler model (Occam's Razor)

# Evaluate on the test set with the best hyper params
modKNN = KNeighborsClassifier(n_neighbors = int(bestParams))
modKNN.fit(x_train,y_train) # train model on training set
preds = modKNN.predict(x_test) # now predict on the test set
testAccKNN = np.mean(y_test == preds)

# =============================================================================
# Decision Tree Stump (max_depth = 1)
# =============================================================================
modStump = DecisionTreeClassifier(max_depth=1) # stump
modStump.fit(x_train, y_train)
preds = modStump.predict(x_test) # now predict on the test set
testAccStump = np.mean(y_test == preds)

# =============================================================================
# Random Guessing
# =============================================================================
np.random.seed(1) # set the seed
preds = np.random.uniform(0,1,len(y_test))
preds = np.where(preds>0.5,1,0)
testAccRandomGuessing = np.mean(y_test == preds)

