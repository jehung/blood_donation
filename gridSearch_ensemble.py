import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
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
score_func = make_scorer(log_loss, greater_is_better=False, needs_proba=True)


def make_feature(data):
    temp = (data['Months since First Donation']-data['Months since Last Donation'])/data['Number of Donations']
    temp -= data['Months since Last Donation']
    data['if'] = abs(temp)

    temp1 = data['If']
    data['log'] = (temp1-temp1.mean())/temp1.std()
    print(data.head())

    temp2 = (data['If'] + (data['Number of Donations'])**2 - 10.2)/data['Months since First Donation']
    data['eureka'] = 1 / (1 + np.exp(-1*temp2))

    data['feature'] = data['eureka'] - data['log']
    #print(data.head())
    return data


def gridSearch_models(data):
    numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation',
                          'If', 'log']

    y = data['Made Donation in March 2007']
    X = data.loc[:, numerical_features]
    print('X', X.head())
    #X_x = PolynomialFeatures(2).fit_transform(X)
    from sklearn import preprocessing
    X_x = preprocessing.normalize(X)
    X_x1 = PolynomialFeatures(2).fit_transform(X_x)
    #X_train, X_test, y_train, y_test = train_test_split(X_x, y, test_size=0.2)
    #print('X_train', X_train.shape)

    
    
    logit = LogisticRegression(random_state=42)
    forest = RandomForestClassifier(random_state=43)
    svc = SVC(probability=True)
    dt = DecisionTreeClassifier(max_features="auto")
    mlp = MLPClassifier(activation='logistic')
    

    # Creating a feature-selection-classifier pipeline

    params = {
        'clf1_pipe__sfs__k_features': [2, 3, 4],
        'clf1_pipe__logit__C': list(np.power(3.0, np.arange(-7, 7))),
        'clf1_pipe__logit__penalty': ['l1', 'l2'],
        'clf1_pipe__logit__fit_intercept': [True, False],
        'svc__kernel': ['linear', 'rbf', 'sigmoid'],
        'svc__C': list(np.power(3.0, np.arange(-15, -5))),
        'svc__gamma': list(np.power(3.0, np.arange(-15, -5))),
        'svc__class_weight': [None, 'balanced'],
        'mlp__activation': ['logistic', 'tanh'],
        'mlp__solver': ['sgd', 'adam'],
        'mlp__learning_rate': ['invscaling', 'adaptive'],
        'mlp__learning_rate_init': [0.001, 0.003, 0.009, 0.03, 0.09, 0.3, 0.9, 1.5],
        'forest__n_estimators': [10, 20, 50, 70, 100, 120, 150],
        'forest__max_features': ['auto', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None],
        'forest__criterion': ['gini', 'entropy'],
        'dt__min_samples_split': [5, 10, 15, 20, 25, 35, 50],
        'dt__max_depth': [3, 4, 5, 6, 7],
        'dt__class_weight': [None, 'balanced']}

    
    sfs1 = SequentialFeatureSelector(logit, 
                                     k_features=4,
                                     forward=True, 
                                     floating=False, 
                                     scoring='neg_log_loss',
                                     verbose=0,
                                     cv=20)

    clf1_pipe = Pipeline([('sfs', sfs1), ('logit', logit)])
    
    eclf = VotingClassifier(estimators=[('clf1_pipe', clf1_pipe), 
                                        ('svc', svc), ('forest', forest), ('dt', dt), ('mlp', mlp)], 
                                        weights=[1, 1, 1,1, 1], voting='soft')
    
    
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=20, scoring='neg_log_loss', n_jobs=-1, verbose=3)
    grid.fit(X_x1, y)
    
    
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))
                 
    print('best choice', grid.best_params_)
    
    eclf = eclf.set_params(**grid.best_params_)
    eclf.fit(X_x1, y).predict_proba(X[[1, 51, 149]])
    
    
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
    test_X_x = preprocessing.normalize(test_X)
    test_X_x1 = PolynomialFeatures(2).fit_transform(test_X)


    predict_proba = eclf.fit(X_x1, y).predict_proba(test_X_x1)
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

