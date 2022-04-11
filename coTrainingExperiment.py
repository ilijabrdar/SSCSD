import pickle
from re import L

import pandas as pd
# from coTraining import co_train, get_score
from cotraining_blum_mitchell import co_train, get_score
from imblearn.over_sampling import SMOTE
import random
from os import listdir
from os.path import isfile, join
import json

BLOB_CK_METRICS = 'data\\god_class\\multi_view\\'
LM_CK_METRICS = 'data\\long_method\\multi_view\\'

# code2vec paths
C2V = 'data\\god_class\\file_code\\'
UNLABELED_C2V = 'data\\god_class\\unlabeled_file_code\\'

C2V_LM = 'data\\long_method\\file_code\\'
C2V_UNLABELED = 'data\\c2v_unlabeled\\'

UNLABELED_JSON_LM = 'data\\long_method\\c2v_unlabeled\\'
UNLABELED_OUTPUT_DIR_LM = 'data\\long-method\\c2v_unlabeled\\'
# code2seq paths
# C2V = '..\\data\\god_class\\c2s\\'
# UNLABELED_C2V = '..\\data\\god_class\\c2s_unlabeled\\'


def pair_ck_with_code2vec(df: pd.DataFrame, cols, labeled=True):
    features = []
    labels = []
    if labeled:
        for _, row in df.iterrows():
            name = str(row['sample_id'].astype(int))
            try:
                with open(f'{C2V}{name}.pkl', 'rb') as f:
                    vec = pickle.load(f)
                    if len(vec.shape) > 1:
                        vec = vec[0]
                    if vec.size != 384:
                        continue
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
                    if len(vec.shape) > 1:
                        vec = vec[0]
                    if vec.size != 384:
                        continue
                    features.append([row[cols].to_list(), vec.tolist()])
            except FileNotFoundError:
                pass

    return features, labels

def pair_ck_with_code2vec_lm(df: pd.DataFrame, cols, labeled = True):
    features = []
    labels = []
    
    if labeled:
        for _, row in df.iterrows():
            name = str(row['sample_id'].astype(int))
            try:
                with open(f'{C2V_LM}{name}.pkl', 'rb') as f:
                    vec = pickle.load(f)
                    if len(vec.shape) > 1:
                        vec = vec[0]
                    if vec.size != 384:
                        continue
                    features.append([row[cols].to_list(), vec.tolist()])
                    labels.append(row['label'])
            except FileNotFoundError:
                pass
    else:
        try:
            for f1 in listdir(UNLABELED_JSON_LM):
                    file_path = join(UNLABELED_JSON_LM, f1)
                    with open(file_path, 'rb') as f:
                        details = pickle.load(f)
                        
                        df_rows = df.loc[df['file'] == details['filePath']]
                        found = False
                        for _, row in df_rows.iterrows():
                            method_details = row['method']
                            method_details_list = method_details.split('/')
                            method_name = method_details_list[0]
                            
                            parameter_details_list = []
                            param_num = 0
                            try:
                                parameter_details_list = method_details_list[1].split('[')
                                param_num = parameter_details_list[0]
                            except Exception as e:
                                print(e)
                            
                            if int(param_num) != details['paramNumber'] or method_name != details['name']: 
                                continue
                            elif parameter_details_list[0] == '0' and details['paramNumber'] == 0 and details['name'] == method_name:
                                features.append([row[cols].to_list(), details['code'].tolist()])
                                found = True
                                break
                                
                            if len(parameter_details_list) == 0:
                                features.append([row[cols].to_list(), details['code'].tolist()])
                                found = True
                                break
                            
                            param_types = parameter_details_list[1].split(']')[0].split(',')

                                
                            has_all_attrs = True
                            for param_type in param_types:
                                type_p = param_type.split('.')
                                if (len(type_p) > 1):
                                    type_p = type_p[len(type_p) - 1]
                                else:
                                    type_p = type_p[0]
                                
                                if type_p not in details['paramTypes']:
                                    has_all_attrs = False
                                    break    
                            if has_all_attrs:
                                features.append([row[cols].to_list(), details['code'].tolist()])   
                                found = True    
                                
                        if found:
                            continue
                             
        except FileNotFoundError:
            pass
        
    return features, labels
                
                

def code2vec_ck_method_exp(smell_type, model, k=30, ratio=2, u=75, p=1, n=3, testing=False):
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

    models = co_train(train, unlabeled, val, model, k=k, ratio=ratio, u=u, p=p, n=n)

    if testing:
        get_score(test, models)


def ck_method_exp(smell_type, model, k=30, ratio=2, u=75, p=1, n=3, testing=False):
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

    models = co_train(train, unlabeled, val, model, k=k, ratio=ratio, u=u, p=p, n=n)

    print(view1)
    print(view2)

    if testing:
        get_score(test, models)

def code2vec_ck_method_experiment_lm(model, k=30, ratio=2, u=75, p=1, n=3, testing=False):
    input_dir = LM_CK_METRICS
    unlabeled_df = pd.read_excel(f'{input_dir}unlabeled.xlsx')
    
    # cols = ['line',	'cbo',	'cboModified',	'fanin', 'fanout',	'wmc',	'rfc',	'loc',	'returnsQty',	'variablesQty',	'parametersQty',	'methodsInvokedQty',
    #         'methodsInvokedLocalQty',	'methodsInvokedIndirectLocalQty',	'loopQty',	'comparisonsQty',	'tryCatchQty',	'parenthesizedExpsQty',	'stringLiteralsQty',	'numbersQty',	'assignmentsQty',
    #         'mathOperationsQty',	'maxNestedBlocksQty',	'anonymousClassesQty',	'innerClassesQty',	'lambdasQty',	'uniqueWordsQty',	'modifiers',	'logStatementsQty']

    cols = ['cbo',	'wmc',	'rfc',	'loc',	'returnsQty',	'variablesQty',	'parametersQty',
            'methodsInvokedQty',	'methodsInvokedLocalQty',	'methodsInvokedIndirectLocalQty',	'loopQty',	'comparisonsQty',	
            'tryCatchQty',	'parenthesizedExpsQty',	'stringLiteralsQty',	'numbersQty',	'assignmentsQty',	'mathOperationsQty',	
            'maxNestedBlocksQty',	'anonymousClassesQty',	'lambdasQty',	'uniqueWordsQty',	'modifiers',	'logStatementsQty']
    train_df = pd.read_excel(f'{input_dir}train.xlsx')
    test_df = pd.read_excel(f'{input_dir}test.xlsx')
    val_df = pd.read_excel(f'{input_dir}validation.xlsx')
    unlabeled_df = pd.read_excel(f'{input_dir}unlabeled.xlsx')
    
    train = pair_ck_with_code2vec_lm(train_df, cols)
    test = pair_ck_with_code2vec_lm(test_df, cols)
    val = pair_ck_with_code2vec_lm(val_df, cols)
    unlabeled = pair_ck_with_code2vec_lm(unlabeled_df, cols, False)[0]
    
    models = co_train(train, unlabeled, val, model, k=k, ratio=ratio, u=u, p=p, n=n)

    if testing:
        get_score(test, models)

if __name__ == '__main__':
    # ck_method_exp(smell_type='blob', model='xgboost', k=80, ratio=2, u=70, p=2, n=5, testing=True)
    # ck_method_exp(smell_type='blob', model='xgboost', k=100, ratio=2, u=70, p=2, n=5)

    # code2vec_ck_method_exp(smell_type='blob', model='svm', k=100, ratio=2, u=70, p=2, n=5)
    # code2vec_ck_method_exp('blob', 'bagging', ratio=.2, testing=True)
    
    code2vec_ck_method_experiment_lm(model='svm', k=100, ratio=2, u=70, p=2, n=5)
    # ck_method_exp(smell_type='long_method', model='svm', k=70, ratio=2, u=70, p=2, n=5, testing=True)
    # ck_method_exp(smell_type='long_method', model='svm', k=80, ratio=2, u=70, p=2, n=5, testing=False)
    