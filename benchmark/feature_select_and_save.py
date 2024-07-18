import numpy as np
from feature_select import *
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel
import experiment_config
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



dataset_path,dataset_name, dirs = experiment_config.get_path_and_name()
num_dataset = len(dataset_path)

def features_select_and_save() :
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)


    for i in tqdm(range(num_dataset), desc="Encoding datasets"):
        print("Encoding dataset: ", dataset_name[i])
        print("Loading dataset: ", dataset_path[i])
        print("Dir name: ", dirs[i])
        logging.info(f"Encoding dataset: {dataset_name[i]}")
        logging.info(f"Loading dataset from: {dataset_path[i]}")
        logging.info(f"Dir name: {dirs[i]}")
        
        df = pd.read_json(dataset_path[i], lines=True)
        texts = df['text'].tolist()
        feature = bert_encode_batch(texts, tokenizer, model, max_length=512, batch_size=32)

        print(feature[0][:2])
        print("Saving features for dataset: ", dataset_name[i], " with shape: ", feature.shape, " to ./feature/" + dataset_name[i] + '_feature.npy')
        logging.info("Saving features for dataset: {} with shape: {} to ./feature/{}_feature.npy".format(dataset_name[i], feature.shape, dataset_name[i]))
        np.save('./feature/' + dataset_name[i] + '_feature.npy', feature)
        try:
            logging.info("Saving features for dataset: %s with shape: %s to %s", 
             dataset_name[i], feature.shape, './data/' + dirs[i] + '/' + dataset_name[i] + '_bert_base_uncased_feature.npy')
            np.save('./data/' + dirs[i] + '/' + dataset_name[i] + '_bert_base_uncased_feature.npy', feature)
        except:
            print("Error saving features")
        print("-----------------------------")



def start_feature_select_and_save():
    features_select_and_save()

    def load_features_and_test():
        df = [pd.read_json(dataset_path[i], lines=True) for i in range(num_dataset)]
        texts = [df[i]['text'].tolist() for i in range(num_dataset)]
        labels = [df[i]['label'].tolist() for i in range(num_dataset)]
        print("len df: ", len(df))
        print("len dataset", num_dataset)
        # #test if the features are saved correctly
        new_feature = [np.load('./data/'+dirs[i]+'/'+dataset_name[i]+'_bert_base_uncased_feature.npy') for i in range(num_dataset)]
        
        for i in range(num_dataset):
            print(dataset_name[i], end=' ')
            print(new_feature[i].shape)

        X_train, X_test, y_train, y_test = [], [], [], []
        for i in range(num_dataset):
            if 'train' in dataset_name[i]:
                X_train.append(new_feature[i])
                y_train.append(labels[i])
            elif 'test' in dataset_name[i]:
                X_test.append(new_feature[i])
                y_test.append(labels[i])

        print("len X_train: ", len(X_train))
        print("len X_test: ", len(X_test))
        print("len y_train: ", len(y_train))
        print("len y_test: ", len(y_test))
        for i in range(len(X_train)):
            print(dataset_name[i], end=' ')
            print(X_train[i].shape, X_test[i].shape, len(y_train[i]), len(y_test[i]))
            
    # try:
    #     load_features_and_test()
    # except:
    #     print("Error loading features")
    load_features_and_test()

start_feature_select_and_save()