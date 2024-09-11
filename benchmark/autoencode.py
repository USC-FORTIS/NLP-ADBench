from feature_select import *
from pyod.utils.data import evaluate_print
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel
import experiment_config
dataset_path,dataset_name = experiment_config.get_path_and_name()
num_dataset = len(dataset_path)
# load dataset
df = [pd.read_json(dataset_path[i], lines=True) for i in range(num_dataset)]
texts = [df[i]['text'].tolist() for i in range(num_dataset)]
labels = [df[i]['label'].tolist() for i in range(num_dataset)]
for i in range(num_dataset):
    print(dataset_name[i], end=' ')
    print(len(texts[i]))

features = [np.load('./feature/'+dataset_name[i]+'_feature.npy') for i in range(num_dataset)]
for i in range(num_dataset):
    print(dataset_name[i], end=' ')
    print(features[i].shape)

# split dataset
X_train, X_test, y_train, y_test = [], [], [], []
for i in range(num_dataset):
    xtrain, xtest, ytrain, ytest = train_test_split(features[i], labels[i], test_size=0.33, random_state=42)
    X_train.append(xtrain)
    X_test.append(xtest)
    y_train.append(ytrain)
    y_test.append(ytest)
for i in range(num_dataset):
    print(dataset_name[i], end='  ')
    print(X_train[i].shape, X_test[i].shape, len(y_train[i]), len(y_test[i]))

from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import evaluate_print
def auto_encoder_benchmark(X_train, X_test, y_train, y_test):
    contamination = 0.1
    # train AutoEncoder detector
    clf_name = 'AutoEncoder'
    clf = AutoEncoder(epochs=30, contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
for i in range(num_dataset):
    print(dataset_name[i]+':')
    auto_encoder_benchmark(X_train[i], X_test[i], y_train[i], y_test[i])
    print()