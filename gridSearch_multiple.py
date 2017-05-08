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
import multiple_models
from sklearn.preprocessing import PolynomialFeatures, scale



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

    models1 = {
        'logit': LogisticRegression(warm_start=True),
        'svc': SVC(probability=True),
        'mlp':  MLPClassifier(activation='logistic'),
        'forest': RandomForestClassifier(),
        'dt': DecisionTreeClassifier(max_features="auto")
    }

    params1 = {
        'logit': {'C': list(np.power(3.0, np.arange(-7, 7))),
                  'penalty': ['l1', 'l2'],
                  'class_weight': [None, 'balanced'],
                  'fit_intercept': [True, False]},
        'svc': {'kernel': ['linear', 'rbf', 'sigmoid'],
                'C': list(np.power(3.0, np.arange(-15, -5))),
                'gamma': list(np.power(3.0, np.arange(-15, -5))),
                'class_weight': [None, 'balanced']},
        'mlp': {'solver': ['sgd', 'adam'],
               'learning_rate': ['invscaling', 'adaptive'],
               'learning_rate_init': [0.001, 0.003, 0.009, 0.03, 0.09, 0.3, 0.9, 1.5]},
        'forest': {'n_estimators': [10, 20, 50, 70, 100, 120, 150],
                   'max_features': ['auto', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None],
                   'criterion': ['gini', 'entropy']},
        'dt': {'min_samples_split': [5, 10, 15, 20, 25, 35, 50],
              'max_depth': [3, 4, 5, 6, 7],
              'class_weight': [None, 'balanced']}
    }


    cv = model_selection.KFold(n_splits=20, shuffle=False, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)

    helper1 = multiple_models.EstimatorSelectionHelper(models1, params1)
    models = helper1.fit(X_x1, y, scoring='neg_log_loss', n_jobs=-1, cv=sss)
    
    a = helper1.score_summary(sort_by='median_score')
    print(a)
    #b = helper1.get_best(models, X_x, y)
    df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'w4', 'w5', 'mean', 'std'))

    i = 0
    for w1 in range(1,6):
        for w2 in range(1,6):
            for w3 in range(1,6):
                for w4 in range(1,6):
                    for w5 in range(1,6):

                        if len(set((w1,w2,w3,w4,w5))) == 1: # skip if all weights are equal
                            continue

                        eclf = VotingClassifier(estimators=models, weights=[w1,w2,w3,w4,w5], voting='soft')
                        scores = cross_validation.cross_val_score(
                                                estimator=eclf,
                                                X=X_x1,
                                                y=y,
                                                cv=5,
                                                scoring='neg_log_loss',
                                                n_jobs=1)

                        df.loc[i] = [w1, w2, w3, w4, w5, scores.mean(), scores.std()]
                        i += 1

    df.sort(columns=['mean', 'std'], ascending=False)
    
    writer = pd.ExcelWriter('output_weights.xlsx')
    df.to_excel(writer, 'Sheet1')

    eclf = VotingClassifier(estimators=models, voting='soft')
    eclf.fit(X_x1, y)

    test_file = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'

    test = pd.read_csv(test_file)
    test1 = make_feature(test)
    test_X = test1.loc[:, numerical_features]
    test_X_x = preprocessing.normalize(test_X)
    test_X_x1 = PolynomialFeatures(2).fit_transform(test_X)


    predict_proba = eclf.predict_proba(test_X_x1)
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

