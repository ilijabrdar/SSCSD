import pandas as pd
import os

LM_DS_PATH = '..\\..\\data\\long_method\\multi_view\\unlabeled.xlsx'
OUTPUT_DIR = '..\\..\\data\\long_method\\methods'
JAR_PATH = '.\\MethodExtractor\\target\\extractor-1.0-SNAPSHOT.jar'

if __name__ == '__main__':
    df = pd.read_excel(LM_DS_PATH)
    df.drop_duplicates(subset=['file'], inplace=True)
    files = df['file']
    for file in files:
        try:
            os.system(f'java -cp {JAR_PATH} extractor.MainApp -file {file} -output {OUTPUT_DIR}')
        except Exception:
            print(file)

