from feature_select import *
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel
import experiment_config

dataset_path,dataset_name = experiment_config.get_path_and_name()
num_dataset = len(dataset_path)
# load dataset

def features_select_and_save() :
    df = [pd.read_json(dataset_path[i], lines=True) for i in range(num_dataset)]
    texts = [df[i]['text'].tolist() for i in range(num_dataset)]
    labels = [df[i]['label'].tolist() for i in range(num_dataset)]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print(dataset_name,dataset_path)
    features = [bert_encode_batch(texts[i], tokenizer, model, max_length=512, batch_size=32) for i in range(num_dataset)]

    for i in range(len(features)):
        print(features[i][0][:10])
        print()

    for i in range(len(features)):
        np.save('./feature/'+dataset_name[i]+'_feature.npy', features[i])



def start_feature_select_and_save():
    features_select_and_save()

    df = [pd.read_json(dataset_path[i], lines=True) for i in range(num_dataset)]
    texts = [df[i]['text'].tolist() for i in range(num_dataset)]
    labels = [df[i]['label'].tolist() for i in range(num_dataset)]
    
    #test if the features are saved correctly
    new_feature = [np.load('./feature/'+dataset_name[i]+'_feature.npy') for i in range(num_dataset)]
    for i in range(num_dataset):
        print(new_feature[i].shape)


    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(num_dataset):
        xtrain, xtest, ytrain, ytest = train_test_split(new_feature[i], labels[i], test_size=0.33, random_state=42)
        X_train.append(xtrain)
        X_test.append(xtest)
        y_train.append(ytrain)
        y_test.append(ytest)
    for i in range(num_dataset):
        print(dataset_name[i], end='')
        print(X_train[i].shape, X_test[i].shape, len(y_train[i]), len(y_test[i]))

start_feature_select_and_save()