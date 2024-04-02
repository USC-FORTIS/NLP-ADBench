from pyod.models.kde import KDE
from pyod.utils.data import evaluate_print
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cupy as cp
#%%
import experiment_config
dataset_path,dataset_name = experiment_config.get_path_and_name()
num_dataset = len(dataset_path)
# dataset_name = ["clickbait_nonclickbait", "Corona_NLP", "movie_review", "sms_spam"]
# dataset_path = ["./data/clickbait_nonclickbait.jsonl", "./data/Corona_NLP.jsonl", "./data/movie_review.jsonl", "./data/sms_spam.jsonl"]
# load dataset
df = [pd.read_json(dataset_path[i], lines=True) for i in range(num_dataset)]
texts = [df[i]['text'].tolist() for i in range(num_dataset)]
labels = [df[i]['label'].tolist() for i in range(num_dataset)]
# dataset_name = ["clickbait_nonclickbait", "Corona_NLP", "movie_review", "sms_spam"]
features = [np.load('./feature/'+dataset_name[i]+'_feature.npy') for i in range(num_dataset)]
X_train, X_test, y_train, y_test = [], [], [], []
for i in range(num_dataset):
    xtrain, xtest, ytrain, ytest = train_test_split(features[i], labels[i], test_size=0.33, random_state=42)
    X_train.append(xtrain)
    X_test.append(xtest)
    y_train.append(ytrain)
    y_test.append(ytest)
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print
def knn_mahalanobis_benchmark(X_train, X_test, y_train, y_test):
    contamination = 0.1
    X_train = cp.array(X_train)
    X_test = cp.array(X_test)
    y_train = cp.array(y_train)
    y_test = cp.array(y_test)
    print('cpdone')
    # train HBOS detector
    clf_name = 'KDE'
    clf = KDE()
    clf.fit(X_train)
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores.get())
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores.get())


for i in range(num_dataset):
    print(i)
    print(dataset_name[i]+':')
    knn_mahalanobis_benchmark(X_train[i], X_test[i], y_train[i], y_test[i])
    print()