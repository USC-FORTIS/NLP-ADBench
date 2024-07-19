import numpy as np
from openai import OpenAI
from openai import BadRequestError
import nltk
from nltk.tokenize import sent_tokenize
import logging

def gpt_encode_batch(texts, model_name, batch_size=32,):
    # The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
    client = OpenAI()
    # client = OpenAI(api_key="you api key")

    features = []
    print(f"Total batches: {(len(texts) + batch_size - 1) // batch_size}")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=model_name,
                input=batch_texts
            )
            batch_features = np.array([item.embedding for item in response.data])
            features.append(batch_features)
        except BadRequestError as e:
            # some data's text is too long, so we need to split it into smaller parts, and then average the embeddings
            logging.error(f"Token limit exceeded, adjusting the batch size..")
            logging.error(f"Error: {e}")
            error_message = str(e)
            requested_tokens = int(error_message.split("requested ")[1].split(" tokens")[0])
            token_limit = 8192
            if requested_tokens > token_limit:
                factor = requested_tokens // token_limit + 1
            # split the text to factor parts
            splited_texts = split_text_into_parts(batch_texts[0], factor)
            
            embeddings = []
            for text in splited_texts:
                response = client.embeddings.create(
                    model=model_name,
                    input=text
                )
                batch_features = np.array([item.embedding for item in response.data])
                embeddings.append(batch_features)
            # average the embeddings
            mean_embeddings = np.mean(embeddings, axis=0)
            features.append(mean_embeddings)
            
            
        if i % 100 == 0:
            print(f"Batch {i // batch_size + 1} completed")
    
    features = np.concatenate(features, axis=0)
    return features


def split_text_into_parts(text, number_of_parts=2):
    nltk.download('punkt')  
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    part_size = total_sentences // number_of_parts
    parts = []
    for i in range(number_of_parts):
        start = i * part_size
        if i == number_of_parts - 1:
            end = total_sentences
        else:
            end = start + part_size
        parts.append(' '.join(sentences[start:end]))
    logging.info(f"Splitting text into {number_of_parts} parts")
    logging.info(f"parts: {parts}")
    return parts