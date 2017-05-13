import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, BaggingClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, scale
from mlxtend.feature_selection import SequentialFeatureSelector




file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train_cleaned.csv'
data = pd.read_csv(file_in)


def make_feature(data):
    temp = (data['Months since First Donation']-data['Months since Last Donation'])/data['Number of Donations']
    temp -= data['Months since Last Donation']
    data['if'] = abs(temp)

    temp1 = data['If']
    return data


def gridSearch_models(data):
    numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation',
                          'If']

    y = data['Made Donation in March 2007']
    X = data.loc[:, numerical_features]
    print('X', X.head())
    #X_x = PolynomialFeatures(2).fit_transform(X)
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    #X_x1 = PolynomialFeatures(2).fit_transform(X_x)
    #X_train, X_test, y_train, y_test = train_test_split(X_x, y, test_size=0.2)
    #print('X_train', X_train.shape)

    sss = StratifiedShuffleSplit(n_splits=700, test_size=0.1)
    
    #logit = LogisticRegression()
    logit = LogisticRegressionCV(
        Cs=list(np.power(3.0, np.arange(-10, 10)))
        , penalty='l2'
        , scoring='neg_log_loss'
        , cv=sss
        , random_state=789
        , max_iter=10000
        , fit_intercept=True
        , solver='liblinear'
        , tol=1e-4
    )

    #forest = RandomForestClassifier(random_state=43)
    svc = SVC(probability=True, class_weight='balanced')
    #dt = DecisionTreeClassifier(max_features="auto")
    mlp = MLPClassifier(activation='relu')

    # Creating a feature-selection-classifier pipeline

    params = {
        #'clf1_pipe__sfs__k_features': [2, 3, 4],
        #'clf1_pipe__logit__C': list(np.power(3.0, np.arange(-5, 5))),
        #'clf1_pipe__logit__penalty': ['l1', 'l2'],
        #'clf1_pipe__logit__fit_intercept': [True, False],
        'clf3_pipe__svc__kernel': ['rbf', 'sigmoid'],
        'clf3_pipe__svc__C': list(np.power(3.0, np.arange(-10, 5))),
        'clf3_pipe__svc__gamma': list(np.power(3.0, np.arange(-10, 5))),
        #'mlp__activation': ['logistic', 'tanh'],
        #'mlp__solver': ['sgd', 'adam'],
        'clf2_pipe__mlp__learning_rate': ['invscaling', 'adaptive'],
        'clf2_pipe__mlp__learning_rate_init': [0.003, 0.009, 0.03, 0.09, 0.3, 0.9, 2.5],
        #'forest__n_estimators': [10, 20, 50, 70, 100, 120, 150],
        #'forest__max_features': ['auto', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None],
        #'forest__criterion': ['gini', 'entropy'],
        #'dt__min_samples_split': [5, 10, 15, 20, 25, 35, 50],
        #'dt__max_depth': [3, 4, 5, 6, 7],
        #'dt__class_weight': [None, 'balanced']
        }

    
    clf1_pipe = Pipeline([('scaler', scaler), ('logit', logit)])
    clf2_pipe = Pipeline([('scaler', scaler), ('mlp', mlp)])
    clf3_pipe = Pipeline([('scaler', scaler), ('svc', svc)])
    
    eclf = VotingClassifier(estimators=[('clf1_pipe', clf1_pipe), 
                                        ('clf2_pipe', clf2_pipe), 
                                        ('clf3_pipe', clf3_pipe)], 
                                        weights=[1, 1, 1], voting='soft')
    
    
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=sss, scoring='neg_log_loss', n_jobs=-1, verbose=3)
    grid.fit(X, y)

    eclf = eclf.set_params(**grid.best_params_)
    bagging = BaggingClassifier(eclf, n_estimator=3)
    results = model_selection.cross_val_score(bagging, X, y, scoring='neg_log_loss', cv=780)
    print(results.mean())
    
                 
    print('best choice', grid.best_params_)
    

    '''
    models1 = {
        'logit': LogisticRegression(warm_start=True),
        'svc': SVC(probability=True),
        'mlp':  MLPClassifier(),
        'forest': RandomForestClassifier(),
        'dt': DecisionTreeClassifier(max_features="auto")
    }
    '''
    
    test_file = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'

    test = pd.read_csv(test_file)
    test1 = make_feature(test)
    test_X = test1.loc[:, numerical_features]
    test_X_x = scaler.fit_transform(test_X)
    #test_X_x1 = PolynomialFeatures(2).fit_transform(test_X)


    predict_proba = bagging.fit(X, y).predict_proba(test_X_x)
    print(predict_proba)
    test_col = pd.DataFrame(predict_proba)
    df_id = test.loc[:, ['ID']]
    test_mid = pd.concat([df_id, test_col], axis=1)
    test_mid.head()
    submission = test_mid.loc[:, ['ID', 1]]
    submission.rename(columns={'ID': '', 1: 'Made Donation in March 2007'}, inplace=True)
    submission.to_csv('submission.csv')

if __name__ == '__main__':
    make_feature(data)
    gridSearch_models(data)

