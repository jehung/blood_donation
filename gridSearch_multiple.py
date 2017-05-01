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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, log_loss
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
import multiple_models

file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train.csv'
data = pd.read_csv(file_in)
score_func = make_scorer(log_loss, greater_is_better=True, needs_proba=True)

def make_feature(data):
    temp = (data['Months since First Donation']-data['Months since Last Donation'])/data['Number of Donations']
    temp -= data['Months since Last Donation']
    data['if'] = abs(temp)

    temp1 = data['If']
    data['Exp_if'] = temp1-temp1.mean()/(temp1.max()-temp1.min())
    return data


def error_function(valset, prob):
    temp = (valset.values*np.log(prob) + (1 - valset.values)*np.log(1 - prob)).sum()
    return -1/len(prob)*temp


def gridSearch_models(data):
    numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation',
                          'if', 'Exp_if']
    y = data['Made Donation in March 2007']
    X = data.loc[:, numerical_features]
    from sklearn.preprocessing import PolynomialFeatures
    X_x = PolynomialFeatures(2).fit_transform(X)
    print('X_x', X_x.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_x, y, test_size=0.2)
    print('X_train', X_train)

    models1 = {
        'logit': LogisticRegression(),
        'svc': SVC(probability=True),
        'gb':  GradientBoostingClassifier(),
        'forest': RandomForestClassifier(),
    }

    params1 = {
        'logit': {'C': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001,
                        0.001, 0.01, 1, 10, 100, 1000, 10000, 100000],
                  'penalty': ['l1', 'l2'],
                  'class_weight': [None, 'balanced'],
                  'fit_intercept': [True, False]},
        'svc': {'kernel': ['linear', 'rbf', 'sigmoid'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],},
        'gb': {'n_estimators': [10, 20, 50, 70, 100, 120, 150],
               'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 15],
               'learning_rate': [0.1, 0.3, 0.6, 0.9, 1.2, 1.5],
               'max_depth': [3, 4, 5, 6, 7]},
        'forest': {'n_estimators': [10, 20, 50, 70, 100, 120, 150],
                   'max_features': ['auto', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None],
                   'criterion': ['gini', 'entropy']}
    }


    #cv = model_selection.KFold(n_splits=20, shuffle=False, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2)

    if __name__ == '__main__':
        helper1 = multiple_models.EstimatorSelectionHelper(models1, params1)
        helper1.fit(X_x, y, scoring=score_func, n_jobs=-1)

    print('Processes done')

make_feature(data)
clf = gridSearch_models(data)