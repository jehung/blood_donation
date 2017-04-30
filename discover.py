import csv
from read_data import read_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import sklearn.preprocessing as preprocessing
import pickle as pickle
import imblearn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import EnsembleVoteClassifier

file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train.csv'

numerical_features = ['Made Donation in March 2007', 'Months since Last Donation', 'Number of Donations']
data_raw = read_data(file_in, numerical_features)
data = featureFormat(data_raw, features=numerical_features)

labels, features = targetFeatureSplit(data)

features_x = preprocessing.PolynomialFeatures().fit_transform(features)


class BaseEstimator_compability(LogisticRegression):
    def predict(self, X):
        return self.predict_proba(X)[:, 1][:, np.newaxis]


def model():
    param_grid = {#"ada__base_estimator__criterion" : ["gini", "entropy"],
        #"ada__base_estimator__splitter" :   ["best", "random"],
        # "ada__n_estimators": [20, 50, 75, 100],
        #'pca__n_components': [1, 2],
        'abC2__base_estimator__max_delta_step': [0.5, 1, 1.5, 2],
        'abC2__base_estimator__max_depth': [2, 3, 4, 5, 6, 7, 8],
        'abC2__base_estimator__min_child_weight': [0.5, 1, 1.5],
        'abC2__base_estimator__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'abC2__base_estimator__subsample': [0.4, 0.5, 0.6],

        # 'eclf__decisiontreeclassifier__min_samples_split': [5, 10, 15, 20, 25, 35, 50],
        # 'eclf__decisiontreeclassifier__class_weight': [None, 'balanced'],
        # 'eclf__decisiontreeclassifier__max_depth': [3, 4, 5, 6, 7],
        # 'eclf__logisticregression__C': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000, 100000],
        # 'eclf__logisticregression__penalty': ['l1', 'l2'],
        # 'eclf__logisticregression__class_weight': [None, 'balanced'],
        #'eclf__adaboostclassifier-1__base_estimator__min_samples_split': [5, 10, 15, 20, 25, 35, 50],
        #'eclf__adaboostclassifier-1__base_estimator__max_depth': [3, 4, 5, 6, 7],
        #'eclf__adaboostclassifier-1__base_estimator__class_weight': [None, 'balanced'],
        #'eclf__adaboostclassifier-2__base_estimator__max_delta_step': [0.5, 1, 1.5, 2],
        #'eclf__adaboostclassifier-2__base_estimator__max_depth': [2, 3, 4, 5, 6, 7, 8],
        #'eclf__adaboostclassifier-2__base_estimator__min_child_weight': [0.5, 1, 1.5],
        #'eclf__adaboostclassifier-2__base_estimator__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #'eclf__adaboostclassifier-2__base_estimator__subsample': [0.4, 0.5, 0.6],

        # 'svc__kernel': ('linear', 'rbf', 'sigmoid'),
        # 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        # 'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        # 'kbest__k': [1, 2],
        # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'mlp__solver': ['lbfgs', 'sgd', 'adam'],
        # 'mlp__learning_rate': ['invscaling', 'adaptive'],
        #  'mlpclassifier__activation': ['logistic', 'tanh', 'relu'],
        #  'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam'],
        #  'mlpclassifier__learning_rate': ['invscaling', 'adaptive'],
        # 'gb__n_estimators': [10, 20, 50, 70, 100, 120, 150],
        # 'gb__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 15],
        # 'gb__learning_rate': [0.1, 0.3, 0.6, 0.9, 1.2, 1.5],
        # 'dt__max_depth': [3, 4, 5, 6, 7]
        # 'kpercentile__percentile': [90],
        # 'kpercentile__percentile': [90],
        # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        # 'tfidf__max_df': [0.4, 0.5, 0.6, 0.7, 0.8],
    }

    robustScale = StandardScaler(with_mean=False)
    pca = decomposition.PCA()
    logistic = LogisticRegression()
    xgboost = xgb.XGBClassifier()
    forest = RandomForestClassifier()
    svc = SVC(probability=True)
    DTC = DecisionTreeClassifier(max_features="auto", class_weight="balanced", max_depth=2)
    nb = GaussianNB()
    et = ExtraTreesClassifier()
    ABC1 = AdaBoostClassifier(base_estimator=DTC)
    ABC2 = AdaBoostClassifier(base_estimator=xgboost)
    ABC3 = AdaBoostClassifier(base_estimator=SVC)
    ABC4 = AdaBoostClassifier(base_estimator=forest)
    ABC5 = AdaBoostClassifier(base_estimator=nb)
    ABC6 = AdaBoostClassifier(base_estimator=et)
    mlp = MLPClassifier(max_iter=100000)
    gen = preprocessing.PolynomialFeatures(2)
    base_estimator = BaseEstimator_compability()
    gb = GradientBoostingClassifier(init=base_estimator)
    # voting = VotingClassifier(estimators=[('ada1', ABC1), ('ada2', ABC2), ('ada3', ABC3), ('ada4', ABC4), ('ada5', ABC5), ('ada6', ABC6)],
    #                          voting='hard')
    #eclf = EnsembleVoteClassifier(clfs=[ABC1, ABC2], voting='soft')
    pipe = Pipeline(steps=[('gen', gen), ('scale', robustScale), ('abC2', ABC2)])
    import pprint as pp
    #pp.pprint(sorted(eclf.get_params().keys()))
    pp.pprint(sorted(pipe.get_params().keys()))

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1)
    # sm = SMOTE()
    # features_res, labels_res = sm.fit_sample(features, labels)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_log_loss', cv=sss)
    grid_search.fit(features, labels)
    clf = grid_search.best_estimator_
    print(grid_search.best_score_)

    return clf


clf = model()


from sklearn import metrics
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, features, labels, cv=sss, scoring='neg_log_loss')
scores.mean()

'''
from tester import test_classifier
print(' ')
# use test_classifier to evaluate the model
# selected by GridSearchCV
print("Tester Classification report")
test_classifier(clf, data_raw, numerical_features)

'''
