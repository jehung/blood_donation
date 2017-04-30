import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train.csv'

numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation']

data = pd.read_csv(file_in)
y = data['Made Donation in March 2007']
X = data.loc[:,numerical_features]


def error_function(valset, prob):
    #temp= myset.iloc[indx]*np.log(prob) + (1-myset.iloc[indx])*np.log(1-prob).sum()
    temp = (valset*np.log(prob) + (1 - valset)*np.log(1 - prob)).sum()
    print('here', temp)
    return -1/len(prob)*temp


def model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2)
    logit = LogisticRegression(solver='liblinear')
    cv = model_selection.KFold(n_splits=20, shuffle=True)
    #sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)

    param_grid = {  # "ada__base_estimator__criterion" : ["gini", "entropy"],
        'C': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000,
              100000],
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 1000, 10000, 100000],
        'fit_intercept': [True, False],
    }

    #clf = GridSearchCV(logit,
    #                   scoring='accuracy',
    #                   cv=sss,
    #                   param_grid=param_grid)

    searchCV = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l1'
        ,scoring='neg_log_loss'
        ,cv=cv
        ,random_state=787
        ,max_iter=10000
        ,fit_intercept=True
        ,solver='liblinear'
        ,tol=1e-4
    )

    searchCV.fit(X_train,y_train)
    print(searchCV.scores_[1].max())


    results = []

    proba = searchCV.predict_proba(X_test)[:,1]
    loss = error_function(y_test, proba)
    print(loss)
    #print(searchCV.predict(X_test))
    #print(y_test.tolist())

'''
    results = []
    for traincv, testcv in cv.split(X_train):
        clf = GridSearchCV(logit,
                           scoring='neg_log_loss',
                           cv=cv,
                           param_grid=param_grid)
        proba = logit.fit(X_train.iloc[traincv, :], y_train.iloc[traincv]).predict_proba(X_train.iloc[testcv,:])[:,1]
        results.append(error_function(y_train, testcv, proba))

    print("Results: " + str(np.array(results).mean()))


'''
clf = model()
