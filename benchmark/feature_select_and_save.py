import numpy as np
from feature_select import *
from feature_select_using_gpt import *
import pandas as pd
from transformers import BertTokenizer, BertModel
import experiment_config
import torch
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_path, dataset_name, dirs = experiment_config.get_path_and_name()
num_dataset = len(dataset_path)

def features_select_and_save_using_bert() :
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    for i in tqdm(range(num_dataset), desc="Encoding datasets"):
        logging.info(f"Encoding dataset: {dataset_name[i]}")
        logging.info(f"Loading dataset from: {dataset_path[i]}")
        logging.info(f"Dir name: {dirs[i]}")
        
        df = pd.read_json(dataset_path[i], lines=True)
        texts = df['text'].tolist()
        feature = bert_encode_batch(texts, tokenizer, model, max_length=512, batch_size=32)

        print(feature[0][:2])
        # logging.info("Saving features for dataset: {} with shape: {} to ./feature/{}_feature.npy".format(dataset_name[i], feature.shape, dataset_name[i]))
        # np.save('./feature/' + dataset_name[i] + '_feature.npy', feature)
        try:
            logging.info("Saving features for dataset: %s with shape: %s to %s", 
             dataset_name[i], feature.shape, './data/' + dirs[i] + '/' + dataset_name[i] + '_bert_base_uncased_feature.npy')
            np.save('./data/' + dirs[i] + '/' + dataset_name[i] + '_bert_base_uncased_feature.npy', feature)
        except:
            logging.error("Error saving features")


def features_select_and_save_using_gpt() :
    if os.environ.get("OPENAI_API_KEY") is None:
        logging.error("OPENAI_API_KEY not found. Please set the environment variable OPENAI_API_KEY")
        return
    MODEL_NAME = 'text-embedding-3-large'

    for i in tqdm(range(num_dataset), desc="GPT Encoding datasets"):
        logging.info(f"GPT Encoding dataset: {dataset_name[i]}")
        logging.info(f"Loading dataset from: {dataset_path[i]}")
        logging.info(f"Dir name: {dirs[i]}")
        
        df = pd.read_json(dataset_path[i], lines=True)
        texts = df['text'].tolist()
        feature = gpt_encode_batch(texts, MODEL_NAME)
        
        print(feature[0][:2])
        
        try:
            logging.info("Saving features for dataset: %s with shape: %s to %s", 
             dataset_name[i], feature.shape, './data/' + dirs[i] + '/' + dataset_name[i] + '_gpt_text-embedding-3-large_feature.npy')
            np.save('./data/' + dirs[i] + '/' + dataset_name[i] + '_gpt_text-embedding-3-large_feature.npy', feature)
        except:
            logging.error("Error saving features")


def start_feature_select_and_save():
    # features_select_and_save_using_bert()

    features_select_and_save_using_gpt()

    def load_features_and_test():
        df = [pd.read_json(dataset_path[i], lines=True) for i in range(num_dataset)]
        texts = [df[i]['text'].tolist() for i in range(num_dataset)]
        labels = [df[i]['label'].tolist() for i in range(num_dataset)]
        print("len df: ", len(df))
        print("len dataset", num_dataset)
        # test if the features are saved correctly
        # For BERT
        # new_feature = [np.load('./data/'+dirs[i]+'/'+dataset_name[i]+'_bert_base_uncased_feature.npy') for i in range(num_dataset)]
        
        # For GPT
        new_feature = [np.load('./data/'+dirs[i]+'/'+dataset_name[i]+'_gpt_text-embedding-3-large_feature.npy') for i in range(num_dataset)]
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

    try:
        load_features_and_test()
    except:
        logging.error("Error loading features")


start_feature_select_and_save()