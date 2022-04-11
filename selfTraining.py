import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import f1_score


def self_train(x, y, model='svm'):
    if model == 'svm':
        base_clf = SVC(C=0.8, kernel='linear', gamma='auto', probability=True, random_state=42)
    elif model == 'random_forest':
        base_clf = RandomForestClassifier(n_estimators=500, max_depth=10, bootstrap=False, random_state=42)
        pass
    elif model == 'xgboost':
        base_clf = GradientBoostingClassifier(learning_rate=.8, max_features='log2', n_estimators=50, random_state=42)
        pass
    elif model == 'bagging':
        base_clf = BaggingClassifier(base_estimator=SVC(C=0.8, kernel='linear', gamma='auto', probability=True, random_state=42), bootstrap=False, random_state=0)
        pass
    else:
        raise ValueError('no such model')


    clf = SelfTrainingClassifier(base_estimator=base_clf, criterion='k_best', k_best=5, max_iter=30, verbose=True)
    # search = RandomizedSearchCV(estimator=clf, param_distributions=params, n_iter=30, scoring='f1', random_state=42)
    # search.fit(x.to_numpy(), y.to_numpy())
    # print(search.best_params_)
    return clf.fit(x, y)


def predict(x, y, model):
    y_pred = model.predict(x)
    f1 = f1_score(y, y_pred)
    print(f1)
