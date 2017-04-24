import csv
from read_data import read_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn import linear_model, decomposition
from sklearn.cluster import KMeans
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
import pickle


with open("data_dict.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    print data_dict

file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train.csv'

numerical_features = ['Made Donation in March 2007', 'Months since Last Donation', 'Number of Donations', 'Total Volume Donated',  'Months since First Donation']

#data_raw = read_data(file_in, numerical_features)
data_raw = data_dict
data = featureFormat(data_raw, features=numerical_features)
labels, features = targetFeatureSplit(data)



def logit():
    param_grid = {#"ada__base_estimator__criterion" : ["gini", "entropy"],
              #"ada__base_estimator__splitter" :   ["best", "random"],
              #"ada__n_estimators": [50, 75, 100],
              'pca__n_components': [1, 2, 3],
              #'dt__min_samples_split': [5, 10, 15, 20, 25, 35, 50],
              #'dt__max_depth': [3, 4, 5, 6, 7],
              #'svc__kernel': ('linear', 'rbf', 'sigmoid'),
              #'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              #'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              #'kbest__k': [1, 2],
              #'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
              #'mlp__solver': ['lbfgs', 'sgd', 'adam'],
              #'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
              #'kpercentile__percentile': [90],
              #'kpercentile__percentile': [90],
              #'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
              #'tfidf__max_df': [0.4, 0.5, 0.6, 0.7, 0.8],
            }
            
    robustScale = Normalizer()        
    pca = decomposition.PCA()
    logistic = linear_model.LogisticRegression()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=sss)
    grid_search.fit(features, labels)
    clf = grid_search.best_estimator_
    print grid_search.best_score_

    return clf
    
    
    
clf = logit()

from tester import test_classifier
print ' '
# use test_classifier to evaluate the model
# selected by GridSearchCV
print "Tester Classification report"
test_classifier(clf, data_dict, features_list)





##############################################################
def get_1_cluster_kmeans():
    ## get K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)


    cluster_labels = kmeans.labels_

    #print cluster_labels
    #print labels

    print float(sum(cluster_labels == labels))/len(cluster_labels)
    #print kmeans.score(features)
    clusters = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    features1 = []
    labels1 = []
    for p in clusters[1]:
        features1.append(data[p])
        labels1.append(labels[p])
    
    return (features1, labels1)
    
    
def get_1_cluster_nn():    
    file_in = 'C:\\Users\\IBM_ADMIN\\PycharmProjects\\Test\\train.csv'

    numerical_features = ['Made Donation in March 2007', 'Months since Last Donation', 'Number of Donations', 'Total Volume Donated',  'Months since First Donation']

    data_raw = read_data(file_in, numerical_features)
    data = featureFormat(data_raw, features=numerical_features)
    labels, features = targetFeatureSplit(data)

    ## get K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)


    cluster_labels = kmeans.labels_

    #print cluster_labels
    #print labels

    print float(sum(cluster_labels == labels))/len(cluster_labels)
    #print kmeans.score(features)
    clusters = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    features1 = []
    labels1 = []
    for p in clusters[1]:
        features1.append(data[p])
        labels1.append(labels[p])
    
    return (features1, labels1)