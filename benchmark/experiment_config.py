# dataset_path = ["./data/clickbait_nonclickbait.jsonl", "./data/Corona_NLP.jsonl", "./data/movie_review.jsonl", "./data/sms_spam.jsonl"]
dataset_path = ["./data/clickbait_nonclickbait.jsonl", "./data/Corona_NLP.jsonl", "./data/movie_review.jsonl"]

def get_path_and_name():
    dataset_name = [path.split('/')[-1].split('.')[0] for path in dataset_path]
    return dataset_path, dataset_name