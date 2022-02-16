from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

colors = ['r', 'g', 'b', 'k']


def co_train(labeled_data, unlabeled_data, test_data, model='svm', view_count=2, learning_speed=5, ratio=1):
    if model == 'svm':
        models = [SVC(C=0.2, kernel='linear', gamma='auto', probability=True, random_state=0) for i in range(view_count)] #ck
        # models = [SVC(C=.2, probability=True, random_state=0), SVC(C=.8, probability=True, random_state=0)] #code2vec
    elif model == 'random_forest':
        # models = [RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5, random_state=0) for i in range(view_count)]
        models = [RandomForestClassifier(min_samples_leaf=5, random_state=0),
                  RandomForestClassifier(min_samples_leaf=5, random_state=0)]
    elif model == 'xgboost':
        # models = [GradientBoostingClassifier(learning_rate=.3, max_features='log2', n_estimators=50, random_state=0) for i in range(view_count)]
        models = [GradientBoostingClassifier(learning_rate=.3, max_features='log2', n_estimators=50, random_state=0),
                  GradientBoostingClassifier(learning_rate=.3, max_features='log2', n_estimators=50, random_state=0)]
    elif model == 'bagging':
        estimator = SVC(C=.2)
        # models = [BaggingClassifier(base_estimator=estimator, bootstrap=False, random_state=0) for i in range(view_count)]
        models = [BaggingClassifier(base_estimator=SVC(C=.2), bootstrap=False, random_state=0),
                  BaggingClassifier(base_estimator=SVC(C=.8), bootstrap=False, random_state=0)]
    else:
        raise ValueError()

    x_views = [[] for i in range(len(models))]
    y_views = [[] for i in range(len(models))]
    labeled_data_x, labeled_data_y = labeled_data
    for j, x in enumerate(labeled_data_x):
        for i in range(view_count):
            x_views[i].append(x[i])
            y_views[i].append(labeled_data_y[j])

    sm = SMOTE(sampling_strategy=0.4, random_state=42)
    for i in range(view_count):
        x_views[i], y_views[i] = sm.fit_resample(x_views[i], y_views[i])

    scores = []
    unlabeled_data_original = len(unlabeled_data)
    while len(unlabeled_data) > 0:
        unlabeled_in_ds = unlabeled_data_original - len(unlabeled_data)
        ul_ratio = unlabeled_in_ds / (unlabeled_in_ds + len(labeled_data_x))
        if ul_ratio >= ratio:
            break
        print(len(unlabeled_data))
        fit(models, x_views, y_views)
        scores.append([get_score(test_data, models), ul_ratio])

        to_add, to_remove = predict(models, unlabeled_data, learning_speed)

        to_remove.sort(reverse=True)
        for idx in to_remove:
            del unlabeled_data[idx]

        for model_idx, x, y in to_add:
            for i in range(view_count):
                if model_idx != i:
                    x_views[i].append(x[i])
                    y_views[i].append(y)

    plot_scores(scores, len(models))
    return models


def fit(models, x_views, y_views):
    for i in range(len(models)):
        models[i].fit(x_views[i], y_views[i])


def predict(models, unlabeled_data, learning_speed):
    to_add = []
    to_remove = set([])
    for i in range(len(models)):
        x_pred_view = []
        for data in unlabeled_data:
            x_pred_view.append(data[i])
        y_pred = models[i].predict_proba(x_pred_view)
        y_pred_filtered = []
        for idx, pred in enumerate(y_pred):
            if pred[0] >= pred[1]:
                y_pred_filtered.append((idx, pred[0] - pred[1], 0))
            else:
                y_pred_filtered.append((idx, pred[1] - pred[0], 1))
        y_pred_filtered.sort(key=lambda sample: sample[1], reverse=True)
        to_add.extend(list(map(lambda sample: (i, unlabeled_data[sample[0]], sample[2]), y_pred_filtered[:learning_speed])))
        to_remove = to_remove.union(set(map(lambda sample: sample[0], y_pred_filtered[:learning_speed])))

    return to_add, list(to_remove)


def get_score(test, models):
    x_views = [[] for i in range(len(models))]
    y_views = [[] for i in range(len(models))]
    test_x, test_y = test
    for j, x in enumerate(test_x):
        for i in range(len(models)):
            x_views[i].append(x[i])
            y_views[i].append(test_y[j])

    scores = []
    for i, model in enumerate(models):
        y = model.predict(x_views[i])
        f1 = f1_score(y_views[i], y)
        scores.append(f1)
        print(f1)

    return scores


def plot_scores(scores, num_scores):
    x = [[] for i in range(num_scores)]
    y = []
    for data in scores:
        score, num = data
        for i, s in enumerate(score):
            x[i].append(s)
        y.append(num)

    plt.figure()
    for i in range(num_scores):
        plt.plot(y, x[i], colors[i])
    plt.show()

