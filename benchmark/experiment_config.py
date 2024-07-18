import os


def collect_jsonl_files_and_directories(base_path):
    files_list = []
    directories_list = []

    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)

        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.jsonl') and ('test_data' in file_name or 'train_data' in file_name):
                    full_path = os.path.join(dir_path, file_name)
                    files_list.append(full_path)
                    directories_list.append(dir_name)

    return files_list, directories_list


base_path = "/home1/yuangang/NLPAD/benchmark/data"


jsonl_files, directories = collect_jsonl_files_and_directories(base_path)

# # dataset_path = ["./data/clickbait_nonclickbait.jsonl", "./data/Corona_NLP.jsonl", "./data/movie_review.jsonl", "./data/sms_spam.jsonl"]
# dataset_path = ["/Users/ken/Desktop/NLP-Anomaly/NLPAD/benchmark/data/agnew.jsonl"]
dataset_path = jsonl_files
print(len(dataset_path))
def get_path_and_name():
    dataset_name = [path.split('/')[-1].split('.')[0] for path in dataset_path]
    return dataset_path, dataset_name, directories