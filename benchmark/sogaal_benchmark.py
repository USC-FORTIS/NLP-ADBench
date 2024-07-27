from pyod.models.so_gaal_new import SO_GAAL
from pyod.utils.data import evaluate_print
import numpy as np
import pandas as pd
import sys
import logging
import os
from sklearn.metrics import average_precision_score

logging.basicConfig(level=logging.INFO)

if len(sys.argv) > 1:
    embedding_method = sys.argv[1]
else:
    logging.error("Please provide an embedding method.")
    sys.exit(1) 

if embedding_method == 'bert':
    embedding_suffix = "_bert_base_uncased_feature.npy"
elif embedding_method == 'gpt':
    embedding_suffix = "_gpt_text-embedding-3-large_feature.npy"
else:
    logging.error("Invalid embedding method. Only 'bert' and 'gpt' are supported.")
    sys.exit(1)

data_dir = './data/'
data_dict = {}

for dir_name in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir_name)
    if os.path.isdir(dir_path):  
        for file in os.listdir(dir_path):
            if file.endswith(embedding_suffix):
                data_type = 'train' if 'train' in file else 'test' if 'test' in file else None
                if data_type:
                    features_path = os.path.join(dir_path, file)
                    features = np.load(features_path)
                    labels_path = os.path.join(dir_path, file.replace(embedding_suffix, '.jsonl'))
                    labels = pd.read_json(labels_path, lines=True)['label'].tolist()
                    
                    if dir_name not in data_dict:
                        data_dict[dir_name] = {'train': {"X": None, "Y": None}, 'test': {"X": None, "Y": None}}
                    data_dict[dir_name][data_type]['X'] = features
                    data_dict[dir_name][data_type]['Y'] = labels

for key, value in data_dict.items():
    logging.info(f"Dataset: {key}")
    for type_key, data in value.items():
        logging.info(f"  {type_key.capitalize()}: {len(data['X'])} samples, {len(data['Y'])} labels")


def so_gaal_benchmark(X_train, X_test, y_train, y_test):
    contamination = 0.1

    # train SO_GAAL detector
    clf_name = 'SO_GAAL'
    clf = SO_GAAL(epoch_num=30, contamination=contamination, verbose=2)

    # Ensure input and target sizes are the same
    if len(X_train) > len(y_train):
        X_train = X_train[:len(y_train)]
    elif len(y_train) > len(X_train):
        y_train = y_train[:len(X_train)]

    clf.fit(X_train)

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    logging.info("On Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
    average_precision = average_precision_score(y_test, y_test_scores)
    logging.info(f"Average Precision: {average_precision}")


logging.info('SO_GAAL_begin')
for dataset_name, data in data_dict.items():
    logging.info("\n\n ")
    logging.info(f"Dataset: {dataset_name}")
    X_train = data['train']['X']
    X_test = data['test']['X']
    y_train = data['train']['Y']
    y_test = data['test']['Y']
    so_gaal_benchmark(X_train, X_test, y_train, y_test)
    logging.info('--------------------------------------------------------')

logging.info('SO_GAAL_done')
