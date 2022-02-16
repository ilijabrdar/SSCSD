import pickle

import pandas as pd
from coTraining import co_train, get_score
from imblearn.over_sampling import SMOTE
import random

BLOB_CK_METRICS = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\multi_view\\'
LM_CK_METRICS = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\multi_view\\'

C2V = '..\\data\\god_class\\file_code\\'
UNLABELED_C2V = '..\\data\\god_class\\unlabeled_file_code\\'

MODEL_PATH = '..\\model\\'


def pair_ck_with_code2vec(df: pd.DataFrame, cols, labeled=True):
    features = []
    labels = []
    if labeled:
        for _, row in df.iterrows():
            name = str(row['sample_id'].astype(int))
            try:
                with open(f'{C2V}{name}.pkl', 'rb') as f:
                    vec = pickle.load(f)
                    features.append([row[cols].to_list(), vec.tolist()])
                    labels.append(row['label'])
            except FileNotFoundError:
                pass
    else:
        for _, row in df.iterrows():
            name = row['class']
            try:
                with open(f'{UNLABELED_C2V}{name}.pkl', 'rb') as f:
                    vec = pickle.load(f)
                    features.append([row[cols].to_list(), vec.tolist()])
            except FileNotFoundError:
                pass

    return features, labels


def code2vec_ck_method_exp(smell_type, model, ratio=1.0, testing=False):
    if smell_type == 'blob':
        input_dir = BLOB_CK_METRICS
    else:
        input_dir = LM_CK_METRICS

    train_df = pd.read_excel(f'{input_dir}train.xlsx')
    test_df = pd.read_excel(f'{input_dir}test.xlsx')
    val_df = pd.read_excel(f'{input_dir}validation.xlsx')
    unlabeled_df = pd.read_excel(f'{input_dir}unlabeled.xlsx')

    # cols = list(train_df.columns)[2:]
    cols = ['totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty',
             'finalMethodsQty',
             'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
             'finalFieldsQty', 'returnQty',	'loopQty', 'tryCatchQty',
             'assignmentsQty', 'variablesQty',
             'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty', 'cbo', 'dit', 'rfc', 'loc']
    train = pair_ck_with_code2vec(train_df, cols)
    test = pair_ck_with_code2vec(test_df, cols)
    val = pair_ck_with_code2vec(val_df, cols)
    unlabeled = pair_ck_with_code2vec(unlabeled_df, cols, labeled=False)[0]

    models = co_train(train, unlabeled, val, model, learning_speed=10, ratio=ratio)

    if testing:
        get_score(test, models)


def ck_method_exp(smell_type, model, ratio=1.0, testing=False):
    if smell_type == 'blob':
        input_dir = BLOB_CK_METRICS
    else:
        input_dir = LM_CK_METRICS

    # # view2 = ['rfc', 'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty',
    # #          'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty',	'synchronizedMethodsQty',
    # #          'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
    # #          'defaultFieldsQty', 'finalFieldsQty', 'returnQty',	'loopQty', 'comparisonsQty', 'tryCatchQty',
    # #          'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty', 'mathOperationsQty', 'variablesQty',
    # #          'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty', 'lambdasQty']
    # view2 = ['rfc', 'totalMethodsQty', 'visibleMethodsQty', 'totalFieldsQty', 'returnQty', 'loopQty', 'tryCatchQty',
    #          'parenthesizedExpsQty', 'stringLiteralsQty', 'variablesQty', 'maxNestedBlocksQty', 'anonymousClassesQty',
    #          'innerClassesQty']

    train_df = pd.read_excel(f'{input_dir}train.xlsx')
    test_df = pd.read_excel(f'{input_dir}test.xlsx')
    val_df = pd.read_excel(f'{input_dir}validation.xlsx')
    unlabeled_df = pd.read_excel(f'{input_dir}unlabeled.xlsx')

    cols = list(train_df.columns)[2:]
    random.seed(42)
    random.shuffle(cols)
    view1, view2 = cols[:len(cols) // 2], cols[len(cols) // 2:]

    train = list(zip(train_df[view1].to_numpy().tolist(), train_df[view2].to_numpy().tolist())), \
                       train_df['label'].to_numpy().tolist()
    test = list(zip(test_df[view1].to_numpy().tolist(), test_df[view2].to_numpy().tolist())), \
            test_df['label'].to_numpy().tolist()
    val = list(zip(val_df[view1].to_numpy().tolist(), val_df[view2].to_numpy().tolist())), \
                       val_df['label'].to_numpy().tolist()
    unlabeled = list(zip(unlabeled_df[view1].to_numpy().tolist(), unlabeled_df[view2].to_numpy().tolist()))

    models = co_train(train, unlabeled, val, model, learning_speed=10, ratio=ratio)

    print(view1)
    print(view2)

    if testing:
        get_score(test, models)


if __name__ == '__main__':
    ck_method_exp(smell_type='blob', model='svm', ratio=0.2, testing=True)
    # ck_method_exp(smell_type='blob', model='svm')

    # code2vec_ck_method_exp('blob', 'bagging')
    # code2vec_ck_method_exp('blob', 'bagging', ratio=.2, testing=True)