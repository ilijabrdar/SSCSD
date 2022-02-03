from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB


def co_train(labeled_data, unlabeled_data, model='svm', view_count=2, learning_speed=5):
    if model == 'svm':
        _model = SVC() #TODO params
    elif model == 'random_forest':
        _model = RandomForestClassifier() #TODO params
    elif model == 'xgboost':
        _model = GradientBoostingClassifier()
    elif model == 'bagging':
        _model = BaggingClassifier()
    elif model == 'naive_bayes':
        _model = GaussianNB()
    else:
        raise ValueError()

    models = [_model] * view_count

    x_views = [[] for i in range(len(models))]
    y_views = [[] for i in range(len(models))]
    for data in labeled_data:
        x, y = data
        for i in range(0, view_count):
            x_views[i].append(x[i])
            y_views[i].append(y)

    while True:
        fit(models, x_views, y_views)
        to_add, to_remove = predict(models, unlabeled_data, learning_speed)

        if len(unlabeled_data) == 0:
            break

        for idx in to_remove:
            del unlabeled_data[idx]

        for i in range(view_count):
            for j in range(view_count):
                if i != j:
                    x, y = to_add[i]
                    x_views[j].extend(x)
                    y_views[j].extend(y)

    return models


def fit(models, x_views, y_views):
    for i in range(len(models)):
        models[i].fit(x_views[i], y_views[i])


def predict(models, unlabeled_data, learning_speed):
    x_pred_view = []
    to_add = []
    to_remove = set([])
    for i in range(len(models)):
        for data in unlabeled_data:
            x_pred_view.append(data[i])
        y_pred = models[i].predict_proba(x_pred_view)
        y_pred_filtered = []
        for idx, pred in enumerate(y_pred):
            if pred[0] >= pred[1]:  # TODO CHECK IF I GUESSED THE ORDER
                y_pred_filtered.append((idx, pred[0], 0))
            else:
                y_pred_filtered.append((idx, pred[1], 1))
        y_pred_filtered.sort(key=lambda sample: sample[1], reverse=True)
        to_add.append(map(lambda sample: (unlabeled_data[sample[0]], sample[2]), y_pred_filtered[:, learning_speed]))
        to_remove.union(set(map(lambda sample: sample[0], y_pred_filtered[:, learning_speed])))

    return to_add, to_remove

