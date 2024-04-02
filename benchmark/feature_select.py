import transformers
import torch
import numpy as np


def bert_encode(texts, tokenizer, model, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
    
    features = outputs[0][:,0,:].numpy()
    return features

def bert_encode_batch(texts, tokenizer, model, max_length, batch_size):
    features = []
    print(f"Total batches: {len(texts)//batch_size+1}")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_features = bert_encode(batch_texts, tokenizer, model, max_length)
        features.append(batch_features)
        # 每20个batch打印一次
        if i % 20 == 0:
            print(f"Batch {i//batch_size+1} done")
    features = np.concatenate(features, axis=0)
    return features


# def feature_select(texts):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#     features = bert_encode_batch(texts, tokenizer, model, max_length=512, batch_size=32)
#     return features

