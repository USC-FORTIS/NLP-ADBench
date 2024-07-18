import numpy as np
from openai import OpenAI

def gpt_encode_batch(texts, model_name, batch_size=32):
    # The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
    client = OpenAI()
    # client = OpenAI(api_key="you api key")

    features = []
    print(f"Total batches: {(len(texts) + batch_size - 1) // batch_size}")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch_texts
        )
        batch_features = np.array([item.embedding for item in response.data])
        features.append(batch_features)
        if i % 20 == 0:
            print(f"Batch {i // batch_size + 1} completed")
    
    features = np.concatenate(features, axis=0)
    return features