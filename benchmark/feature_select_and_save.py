import numpy as np
from numpy.lib import format as npformat
from feature_select import *
from feature_select_using_gpt import *
import pandas as pd
from transformers import BertTokenizer, BertModel
import experiment_config
import torch
from tqdm import tqdm
import logging
import os
import time



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_path, dataset_name, dirs = experiment_config.get_path_and_name()
num_dataset = len(dataset_path)

def features_select_and_save_using_bert() :
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    for i in tqdm(range(num_dataset), desc="Encoding datasets"):
        save_path = './data/' + dirs[i] + '/' + dataset_name[i] + '_bert_base_uncased_feature.npy'
        if os.path.exists(save_path):
            logging.info(f"Feature file already exists for {dataset_name[i]}, skipping...")
            continue
        
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



def append_to_npy(filename, data):
    try:
        if not os.path.exists(filename):
            np.save(filename, data)
            logging.info(f"Created new file: {filename}")
            return

        with open(filename, 'rb+') as f:
            # Read the header of the .npy file
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
            
            # Seek to the end of the file
            f.seek(0, 2)
            
            # Write the new data
            np.lib.format.write_array(f, np.asanyarray(data), version=(1, 0))
            
            # Update the header with the new shape
            f.seek(0)
            new_shape = (shape[0] + len(data), *shape[1:])
            header = np.lib.format.header_data_from_array_1_0(np.empty(new_shape, dtype=dtype))
            np.lib.format.write_array_header_1_0(f, header)
        
        logging.info(f"Successfully appended data to {filename}")
    except Exception as e:
        logging.error(f"Error appending to {filename}: {str(e)}")
        raise

def features_select_and_save_using_gpt() :
    if os.environ.get("OPENAI_API_KEY") is None:
        logging.error("OPENAI_API_KEY not found. Please set the environment variable OPENAI_API_KEY")
        return
    MODEL_NAME = 'text-embedding-3-large'

    for i in tqdm(range(num_dataset), desc="GPT Encoding datasets"):
        save_path = './data/' + dirs[i] + '/' + dataset_name[i] + '_gpt_text-embedding-3-large_feature.npy'
        jsonl_path = './data/' + dirs[i] + '/' + dataset_name[i] + ".jsonl"
        
        start_i = 0
        if os.path.exists(save_path):
            logging.info(f"Feature file {save_path} exists for {dataset_name[i]}, checking consistency...")
            
            # Compare row counts
            df = pd.read_json(jsonl_path, lines=True)
            jsonl_row_count = len(df)
            feature_row_count = np.load(save_path).shape[0]
            # print shape
            logging.info(f"JSONL's Shape:{df.shape},feature's shape:{np.load(save_path).shape}")
            
                        
            if jsonl_row_count == feature_row_count:
                logging.info(f"Row counts match for {dataset_name[i]}. Skipping processing.")
                continue
            else:
                logging.warning(f"Row count mismatch for {dataset_name[i]}. JSONL: {jsonl_row_count}, Feature: {feature_row_count}")
                logging.info(f"Deleting existing feature file and reprocessing.")
                # os.remove(save_path)
                if ("N24News_train_data" in dataset_name[i]) or ("N24News_test_data" in dataset_name[i]):
                    start_i = feature_row_count - 1
                else:
                    raise
        logging.info(f"GPT Encoding dataset: {dataset_name[i]}")
        logging.info(f"Loading dataset from: {dataset_path[i]}")
        logging.info(f"Dir name: {dirs[i]}")
        
        df = pd.read_json(dataset_path[i], lines=True)
        texts = df['text'].tolist()
        batch_size = 2048
        if "email_spam_train_data" in dataset_name[i]:
            batch_size = 1
            logging.info(f"Setting batch size to 1 for {dataset_name[i]}")
        
        
        total_processed = start_i + 1
        logging.info(f"Total batches: {(len(texts) - start_i -1 + batch_size - 1) // batch_size}")
        for j in range(start_i, len(texts), batch_size):
            batch_texts = texts[j:j+batch_size]
            batch_features = gpt_encode_batch(batch_texts, MODEL_NAME, batch_size)
            
            # Append features to file
            append_to_npy(save_path, batch_features)
            total_processed += len(batch_features)
            
            if (j // batch_size + 1) % 10 == 0:
                logging.info(f"Batch {j // batch_size + 1} completed. Total processed: {total_processed}")
                logging.info(f"feature shape now: {np.load(save_path).shape}")
        logging.info(f"Completed processing and saving features for {dataset_name[i]}")
        logging.info(f"Total processed samples: {total_processed}")

        # Final check
        final_feature_count = np.load(save_path).shape[0]
        logging.info(f"Final feature count for {dataset_name[i]}: {final_feature_count}")
        if final_feature_count != len(texts):
            logging.error(f"Final row count mismatch for {dataset_name[i]}. JSONL: {len(texts)}, Feature: {final_feature_count}")
        else:
            logging.info(f"Successfully processed {dataset_name[i]}. Row counts match.")

        # feature = gpt_encode_batch(texts, MODEL_NAME, batch_size)
        
        
        # try:
        #     logging.info("Saving features for dataset: %s with shape: %s to %s", 
        #      dataset_name[i], feature.shape, './data/' + dirs[i] + '/' + dataset_name[i] + '_gpt_text-embedding-3-large_feature.npy')
        #     np.save('./data/' + dirs[i] + '/' + dataset_name[i] + '_gpt_text-embedding-3-large_feature.npy', feature)
        # except:
        #     logging.error("Error saving features")


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