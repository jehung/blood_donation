import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit
import collections
from sklearn.metrics import make_scorer, log_loss



score_func = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
final = []

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.final_models = collections.defaultdict(dict)



    def fit(self, X, y, cv=None, n_jobs=1, verbose=1, scoring=None, refit=True):
        best_score = -20
        best_choice = None
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]

            gs = GridSearchCV(model, params, cv=20, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs
            final.append((key, gs.best_estimator_))
            print(gs.best_estimator_)
            
        return final



    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': scores.min(),
                 'max_score': scores.max(),
                 'mean_score': scores.mean(),
                 'std_score': scores.std(),
                 'median_score': np.median(scores)
            }

            return pd.Series({**params, **d})

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values(by=[sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'median_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        writer = pd.ExcelWriter('output.xlsx')
        df[columns].to_excel(writer, 'Sheet1')

        return df[columns]

    def get_best(self, final, X, y):
        results = []
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        for f in final:
            for traincv, testcv in sss.split(X, y):
                proba = f.fit(X[traincv,:],y.iloc[traincv]).predict_proba(X[testcv,:])[:,1]
                pred = f.fit(X[traincv,:],y.iloc[traincv]).predict(X[testcv,:])
                #print(np.log(proba))
                #results.append(error_function(y_train.iloc[testcv], proba))
                results.append(log_loss(y.iloc[testcv], pred))
                print(f)
                print(log_loss(y.iloc[testcv], pred))
                print('min', min(results))
                print('max', max(results))
        print("Results: ", results)

        #writer = pd.ExcelWriter('finaloutput.xlsx')
        #final[columns].to_excel(writer, 'Sheet1')
