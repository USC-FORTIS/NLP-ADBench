import transformers
import torch
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def bert_encode(texts, tokenizer, model, max_length):
    input_ids = []
    attention_masks = []
    model.to(device)

    for text in texts:
        # encoded_dict = tokenizer.encode_plus(
        #     text,
        #     add_special_tokens=True,
        #     max_length=max_length,
        #     pad_to_max_length=True,
        #     return_attention_mask=True,
        #     return_tensors='pt',
        # )
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',  # Updated padding argument
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # 移动编码后的数据到指定设备
        input_ids.append(encoded_dict['input_ids'].to(device))
        attention_masks.append(encoded_dict['attention_mask'].to(device))
    
    # 使用torch.cat在维度0上连接列表中的所有tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
    
    features = outputs[0][:,0,:].cpu().numpy()  # 将特征移动回CPU并转换为NumPy数组
    return features


def bert_encode_batch(texts, tokenizer, model, max_length, batch_size):
    features = []
    print(f"Total batches: {len(texts)//batch_size+1}")

    for i in tqdm(range(0, len(texts), batch_size), desc="bert_encode_batch"):
        batch_texts = texts[i:i+batch_size]
        batch_features = bert_encode(batch_texts, tokenizer, model, max_length)
        features.append(batch_features)
        if i % 20 == 0:
            print(f"Batch {i//batch_size+1} done")
    features = np.concatenate(features, axis=0)
    return features


# def feature_select(texts):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#     features = bert_encode_batch(texts, tokenizer, model, max_length=512, batch_size=32)
#     return features

