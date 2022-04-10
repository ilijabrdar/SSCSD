from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import random

colors = ['r', 'g', 'b', 'k']


def co_train(labeled_data, unlabeled_data, test_data, model='svm', view_count=2, k=30, ratio=2, u=75, p=1, n=3):
    if model == 'svm':
        # models = [SVC(C=0.2, kernel='linear', gamma='auto', probability=True, random_state=0) for i in range(view_count)] #ck
        # models = [SVC(C=.2, probability=True, random_state=0), SVC(C=100, probability=True, random_state=0)] #code2vec
        models = [SVC(C=.2, probability=True, random_state=0), SVC(C=.8, probability=True, random_state=0)]
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
    iter_count = 0
    while iter_count < k and len(unlabeled_data) > 0:
        print(iter_count)
        fit(models, x_views, y_views)
        scores.append([get_score(test_data, models), iter_count])

        random.seed(iter_count)
        u_prim = random.sample(unlabeled_data, u)
        to_add, to_remove = predict(models, u_prim, p, n)

        # to_remove.sort(reverse=True)
        # for idx in to_remove:
        #     del unlabeled_data[idx]

        for x, y in to_add:
            for i in range(view_count):
                x_views[i].append(x[i])
                y_views[i].append(y)
        u = ratio * (p + n)
        iter_count += 1

    plot_scores(scores, len(models) + 1)
    return models


def fit(models, x_views, y_views):
    for i in range(len(models)):
        models[i].fit(x_views[i], y_views[i])


def predict(models, unlabeled_data, p, n):
    to_add = []
    to_remove = set([])
    for i in range(len(models)):
        x_pred_view = []
        for data in unlabeled_data:
            x_pred_view.append(data[i])
        y_pred = models[i].predict_proba(x_pred_view)
        positive, negative = [], []
        for idx, pred in enumerate(y_pred):
            if pred[0] >= pred[1]:
                negative.append((idx, pred[0] - pred[1], 0))
            else:
                positive.append((idx, pred[1] - pred[0], 1))
        positive.sort(key=lambda sample: sample[1], reverse=True)
        negative.sort(key=lambda sample: sample[1], reverse=True)
        to_add.extend(list(map(lambda sample: (unlabeled_data[sample[0]], sample[2]), positive[:p] + negative[:n])))
        to_remove = to_remove.union(set(map(lambda sample: sample[0], positive[:p] + negative[:n])))

    return to_add, list(to_remove)


def get_score(test, models):
    x_views = [[] for i in range(len(models))]
    test_x, y_views = test
    for j, x in enumerate(test_x):
        for i in range(len(models)):
            x_views[i].append(x[i])

    scores = []
    y_prob = [[] for i in range(len(models))]
    for i, model in enumerate(models):
        y = model.predict(x_views[i])
        f1 = f1_score(y_views, y)
        y_prob[i] = model.predict_proba(x_views[i])
        scores.append(f1)
        print(f1)

    y = []
    for i in range(len(y_prob[0])):
        prob_neg = y_prob[0][i][0] * y_prob[1][i][0]
        prob_pos = y_prob[0][i][1] * y_prob[1][i][1]
        y.append(1) if prob_pos > prob_neg else y.append(0)
    f1_comb = f1_score(y_views, y)
    print(f1_comb)
    scores.append(f1_comb)

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

