import pandas as pd

BLOB_CK_TRAIN = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\multi_view\\train.xlsx'
BLOB_CK_TEST = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\multi_view\\test.xlsx'
BLOB_FILES = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\files\\'
BLOB_OUTPUT_DIR = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\file_code\\'

LM_CK_TRAIN = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\multi_view\\train.xlsx'
LM_CK_TEST = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\multi_view\\test.xlsx'
LM_FILES = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\files\\'
LM_OUTPUT_DIR = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\file_code\\'


def code2vec(smell_type):
    if smell_type == 'blob':
        input_data_train = BLOB_CK_TRAIN
        input_data_test = BLOB_CK_TEST
        input_files = BLOB_FILES
        output = BLOB_OUTPUT_DIR
    elif smell_type == 'long_method':
        input_data_train = LM_CK_TRAIN
        input_data_test = LM_CK_TEST
        input_files = LM_FILES
        output = LM_OUTPUT_DIR
    else:
        raise ValueError()

    df_train = pd.read_excel(input_data_train)
    df_test = pd.read_excel(input_data_test)
    samples_train = df_train['sample_id']
    samples_test = df_test['sample_id']

    not_found_train, not_found_test = [], []

    with open(output + 'train.txt', 'w') as of:
        for idx, sample in enumerate(samples_train):
            try:
                with open(f'{input_files}{sample}.java', 'r') as f:
                    of.write(f.read().replace('\n', ' '))
                    if idx < len(samples_train) - 1:
                        of.write('\n')
            except FileNotFoundError:
                not_found_train.append(sample)

    with open(output + 'test.txt', 'w') as of:
        for idx, sample in enumerate(samples_test):
            try:
                with open(f'{input_files}{sample}.java', 'r') as f:
                    of.write(f.read().replace('\n', ''))
                    if idx < len(samples_test) - 1:
                        of.write('\n')
            except FileNotFoundError:
                not_found_test.append(sample)

    print(not_found_train)
    print(not_found_test)
    # df_train[~df_train['sample_id'].isin(not_found_train)].to_excel(input_data_train, index=False)
    # df_test[~df_test['sample_id'].isin(not_found_test)].to_excel(input_data_test, index=False)


if __name__ == '__main__':
    code2vec(smell_type='long_method')

