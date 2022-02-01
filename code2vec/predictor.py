import pickle
import numpy as np
from extractor import Extractor
import os

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'

BLOB_FILES = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\files\\'
BLOB_OUTPUT_DIR = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\god_class\\file_code\\'

LM_FILES = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\files\\'
LM_OUTPUT_DIR = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\file_code\\'


class Predictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def predict(self, smell_type='blob'):
        if smell_type == 'blob':
            input_files = BLOB_FILES
            output = BLOB_OUTPUT_DIR
        elif smell_type == 'long_method':
            input_files = LM_FILES
            output = LM_OUTPUT_DIR
        else:
            raise ValueError()

        errors = []
        for sample in os.listdir(input_files):
            input_filename = f'{input_files}{sample}'
            try:
                predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
            except ValueError as e:
                errors.append(sample)
                print(e)
                continue

            raw_prediction_results = self.model.predict(predict_lines)
            vectors = []
            for raw_prediction in raw_prediction_results:
                vectors.append(raw_prediction.code_vector)

            vectors = np.array(vectors)
            mean = vectors.mean(axis=0)
            name = sample.split('.')[0]
            with open(f'{output}{name}.pkl', 'wb') as f:
                pickle.dump(mean, f)

        print(errors)
