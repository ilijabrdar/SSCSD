import pandas as pd
from sklearn.preprocessing import StandardScaler

BLOB_METRICS = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\MLCQ_metrics_blob.xlsx'
BLOB_OUTPUT_DIR = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\multi_view\\'

LM_METRICS = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\MLCQ_metrics_LM_multicat.xlsx'
LM_OUTPUT_DIR = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\multi_view\\'


def z_normalize(df: pd.DataFrame, smell_type='blob'):
    if smell_type == 'blob':
        cols_to_norm = df.columns.difference(['parts', 'label', 'sample_id', 'type', 'from_project', 'tcc', 'lcc'])
    else:
        cols_to_norm = df.columns.difference(['parts', 'label', 'from_project', 'sample_id', 'constructor', 'hasJavaDoc'])
    scale = StandardScaler()
    df[cols_to_norm] = scale.fit_transform(df[cols_to_norm])
    return df


def drop_cols(df: pd.DataFrame, smell_type='blob'):
    if smell_type == 'blob':
        cols_to_drop = ['parts', 'type', 'from_project']
    else:
        cols_to_drop = ['parts', 'from_project']
        # df = df[df['constructor'] == 0]             # drop rows referring to constructors
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
    df['label'] = df['label'].apply(lambda x: 0 if x == 'none' else 1)
    df_train = df[df['parts'] == 'train']
    df_test = df[df['parts'] == 'test']

    drop_cols(df_train, smell_type).to_excel(f'{output_dir}train.xlsx', index=False)
    drop_cols(df_test, smell_type).to_excel(f'{output_dir}test.xlsx', index=False)


if __name__ == '__main__':
    process_metrics(smell_type='long_method')


