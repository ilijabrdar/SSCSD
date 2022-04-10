import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def self_train(x, y, model='svm'):
    if model == 'svm':
        base_clf = SVC(C=.2, kernel='linear', gamma='auto', probability=True, random_state=42)
        pass
    elif model == 'random_forest':
        pass
    elif model == 'xgboost':
        pass
    elif model == 'bagging':
        pass
    else:
        raise ValueError('no such model')

    clf = SelfTrainingClassifier(base_estimator=base_clf, criterion='k_best', k_best=5, max_iter=60)
    return clf.fit(x, y)


def predict(x, y, model):
    y_pred = model.predict(x)
    f1 = f1_score(y, y_pred)
    print(f1)
