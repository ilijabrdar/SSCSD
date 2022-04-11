import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from selfTraining import self_train, predict


BLOB_DATASET_PATH = '..\\data\\god_class\\multi_view\\'


def extract_features_and_labels(df, cols):
    return df[cols], df['label']


def train(model='svm', smell_type='blob'):
    if smell_type == 'blob':
        input_dir = BLOB_DATASET_PATH
        cols = ['totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty',
                'finalMethodsQty',
                'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
                'finalFieldsQty', 'returnQty', 'loopQty', 'tryCatchQty',
                'assignmentsQty', 'variablesQty',
                'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty', 'cbo', 'dit', 'rfc', 'loc']
    elif smell_type == 'long_method':
        input_dir = ''
        cols = []
        pass
    else:
        raise ValueError('Smell type can either be blob or long_method')

    train_df = pd.read_excel(f'{input_dir}train.xlsx')
    val_df = pd.read_excel(f'{input_dir}validation.xlsx')
    test_df = pd.read_excel(f'{input_dir}test.xlsx')
    unlabeled_df = pd.read_excel(f'{input_dir}unlabeled.xlsx')

    train_x, train_y = extract_features_and_labels(train_df, cols)
    val_x, val_y = extract_features_and_labels(val_df, cols)
    test_x, test_y = extract_features_and_labels(test_df, cols)
    unlabeled_x, unlabeled_y = unlabeled_df[cols], pd.Series(data=np.array([-1] * unlabeled_df.shape[0]))

    sm = SMOTE(sampling_strategy=0.4, random_state=42)
    train_x, train_y = sm.fit_resample(train_x, train_y)

    train_x, train_y = pd.concat([train_x, unlabeled_x]), pd.concat([train_y, unlabeled_y])
    # self_train(train_x, train_y, model)
    clf = self_train(train_x, train_y, model)
    predict(val_x, val_y, clf)
    predict(test_x, test_y, clf)


if __name__ == '__main__':
    train(model='bagging', smell_type='blob')