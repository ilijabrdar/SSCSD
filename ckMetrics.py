import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BLOB_METRICS = '..\\data\\god_class\\MLCQ_metrics_blob.xlsx'
BLOB_OUTPUT_DIR = '..\\data\\god_class\\multi_view\\'

LM_METRICS = '..\\data\\long_method\\MLCQ_metrics_LM_multicat.xlsx'
LM_OUTPUT_DIR = '..\\data\\long_method\\multi_view\\'

UNLABELED_PROJ_ROOT = '..\\data\\unlabeled\\'
CK_OUTPUT = '..\\data\\unlabeled_ck_metrics\\'
CK_JAR_PATH = '..\\ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar'

TRAIN_SET = 'data\\long_method\\multi_view\\train.xlsx'
VALIDATION_SET = 'data\\long_method\\multi_view\\'
pd.options.mode.chained_assignment = None


def z_normalize(df: pd.DataFrame, smell_type='blob'):
    if smell_type == 'blob':
        df = df.rename({'RM_LOC': 'loc', 'RM_WMC': 'wmc', 'RM_LCOM5': 'lcom*'}, axis=1)
        cols_to_norm = df.columns.difference(['parts', 'label', 'sample_id', 'type', 'from_project', 'tcc', 'lcc'])
        df = df[df['type'] == 1]  # 1 - class 2 - inner class 3 - interface
    else:
        cols_to_norm = df.columns.difference(['parts', 'label', 'from_project', 'sample_id', 'constructor', 'hasJavaDoc'])
    scale = StandardScaler()
    df[cols_to_norm] = scale.fit_transform(df[cols_to_norm])
    return df


def z_normalize_unlabeled(df: pd.DataFrame, smell_type='blob'):
    if smell_type == 'blob':
        cols_to_norm = df.columns.difference(['file', 'class', 'type', 'tcc', 'lcc', 'lcom*'])
        df[['tcc', 'lcc']] = df[['tcc', 'lcc']].fillna(-1)
        df[['lcom*']] = df[['lcom*']].fillna(0.5)
        df['type'] = df['type'].astype('category').cat.codes
        df = df[df['type'] == 1]  # 1 - class 0 - anon 3 - inner class 4 - interface 2 -enum
    else:
        cols_to_norm = df.columns.difference(['file', 'class', 'method', 'constructor', 'hasJavaDoc'])
        df['constructor'] = df['constructor'].astype('category').cat.codes
    scale = StandardScaler()
    df[cols_to_norm] = scale.fit_transform(df[cols_to_norm])
    return df


def drop_cols(df: pd.DataFrame, smell_type='blob'):
    if smell_type == 'blob':
        cols_to_drop = ['parts', 'type', 'from_project', 'RM_ATFD', 'RM_NADC']
    else:
        cols_to_drop = ['parts', 'from_project']
        # df = df[df['constructor'] == 0]             # drop rows referring to constructors
    df['label'] = df['label'].apply(lambda x: 0 if x == 'none' else 1)
    return df.drop(cols_to_drop, axis=1)


def process_metrics(smell_type='blob'):
    if smell_type == 'blob':
        input_dir = BLOB_METRICS
        output_dir = BLOB_OUTPUT_DIR
    elif smell_type == 'long_method':
        input_dir = LM_METRICS
        output_dir = LM_OUTPUT_DIR
    else:
        raise ValueError()

    df = pd.read_excel(input_dir)
    df = z_normalize(df, smell_type)
    # df['label'] = df['label'].apply(lambda x: 0 if x == 'none' else 1)
    df_train = df[df['parts'] == 'train']
    df_test = df[df['parts'] == 'test']

    df_train, df_valid = train_test_split(df_train, test_size=.25, random_state=0, stratify=df_train['label']) #42

    drop_cols(df_train, smell_type).to_excel(f'{output_dir}train.xlsx', index=False)
    drop_cols(df_test, smell_type).to_excel(f'{output_dir}test.xlsx', index=False)
    drop_cols(df_valid, smell_type).to_excel(f'{output_dir}validation.xlsx', index=False)


def split_validation(smell_type='blob'):
    if smell_type == 'blob':
        input_dir = BLOB_METRICS
        output_dir = BLOB_OUTPUT_DIR
    elif smell_type == 'long_method':
        input_dir = TRAIN_SET
        output_dir = VALIDATION_SET
    else:
        raise ValueError()
    
    df = pd.read_excel(input_dir)
    df_train, df_valid = train_test_split(df, test_size=.25, random_state=0, stratify=df['label']) #42
    df_valid.to_excel(f'{output_dir}validation.xlsx', index=False)
    df_train.to_excel(f'{output_dir}train.xlsx', index=False)

    
def generate_ck_metrics():
    for dir_name in os.listdir(UNLABELED_PROJ_ROOT):
        os.mkdir(CK_OUTPUT + dir_name)
        # os.chdir(CK_OUTPUT + dir_name)
        os.system(f'java -jar {CK_JAR_PATH} {UNLABELED_PROJ_ROOT + dir_name} true 0 False {CK_OUTPUT + dir_name}\\ test\\ tests\\ docs\\')


def process_unlabeled_metrics(smell_type='blob'):
    if smell_type == 'blob':
        filename = 'class.csv'
        output = BLOB_OUTPUT_DIR
    elif smell_type == 'long_method':
        filename = 'method.csv'
        output = LM_OUTPUT_DIR
    else:
        raise ValueError()

    df = None
    for root, dirs, files in os.walk(CK_OUTPUT):
        for file in files:
            if file == filename:
                tmp_df = pd.read_csv(os.path.join(root, file))
                if df is None:
                    df = tmp_df
                else:
                    df = pd.concat([df, tmp_df])

    df = z_normalize_unlabeled(df, smell_type)

    if df is not None:
        df.to_excel(f'{output}unlabeled.xlsx', index=False)


if __name__ == '__main__':
    # generate_ck_metrics()
    # process_unlabeled_metrics(smell_type='blob')
    # process_metrics(smell_type='blob')
    split_validation(smell_type = 'long_method')

