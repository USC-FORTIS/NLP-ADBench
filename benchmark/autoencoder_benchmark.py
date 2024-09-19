from pyod.models.auto_encoder import AutoEncoder
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

from sklearn.preprocessing import StandardScaler, normalize

def clean_features(X):
    # Replace NaNs with 0, positive infinity with a large number, negative infinity with a small number
    X = np.nan_to_num(X, nan=0.0, posinf=1e38, neginf=-1e38)

    # Clip the values to the maximum and minimum representable by float32
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    X = np.clip(X, min_float32, max_float32)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to float32
    X = X.astype(np.float32)
    return X

def clean_labels(y):
    # For labels, just ensure they are a proper float32 array and replace NaNs or Infs if necessary
    y = np.nan_to_num(y, nan=0.0, posinf=1e38, neginf=-1e38)
    y = y.astype(np.float32)
    return y

def auto_encoder_benchmark(X_train, X_test, y_train, y_test):
    contamination = 0.1
    # train AutoEncoder detector
    clf_name = 'AutoEncoder'
    clf = AutoEncoder(epoch_num=30, contamination=contamination)
    clf.fit(X_train)

    # Because ONLY normal data is used for training, ROC cn't be calculated (Only one class present in y_true. ROC AUC score is not defined in that case.)
    # get the prediction labels and outlier scores of the training data
    # y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    # print("\nOn Training Data:")
    # evaluate_print(clf_name, y_train, y_train_scores)
    logging.info("On Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
    average_precision = average_precision_score(y_test, y_test_scores)
    logging.info(f"Average Precision: {average_precision}")


logging.info('AE_begin')
for dataset_name, data in data_dict.items():
    logging.info("\n\n ")
    logging.info(f"Dataset: {dataset_name}")
    X_train = data['train']['X']
    X_test = data['test']['X']
    y_train = data['train']['Y']
    y_test = data['test']['Y']
    X_train = clean_features(X_train)
    X_test = clean_features(X_test)
    y_train =  clean_labels(y_train)
    y_test = clean_labels(y_test)
    auto_encoder_benchmark(X_train, X_test, y_train, y_test)
    logging.info('--------------------------------------------------------')

    
    
logging.info('AE_end')