import os
import json
import pickle

import fire
import numpy as np
from scipy import sparse

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def load_split(data_dir, human_source, computer_source, split, n=np.inf):
    human = _load_split(data_dir, human_source, split, n=n//2)
    computer = _load_split(data_dir, computer_source, split, n=n//2)
    texts = human+computer
    labels = [0]*len(human)+[1]*len(computer)
    return texts, labels

def get_path(dir, source, name):
    return os.path.join(dir, f'{source}.{name}')

def dump_model(model_dir, source, name, model):
    serializer = pickle
    path = get_path(model_dir, source, name)
    with open(path, 'wb') as f:
        serializer.dump(model, f)

def load_model(model_dir, source, name):
    serializer = pickle
    path = get_path(model_dir, source, name)
    with open(path, 'rb') as f:
        return serializer.load(f)

def has_model(model_dir, source, name):
    return os.path.exists(get_path(model_dir, source, name))

# adapted from: https://github.com/openai/gpt-2-output-dataset
def main(
    human_source,
    computer_source,
    data_dir='./data',
    model_dir='./model',
    log_dir='./log',
    n_train=18000,
    n_valid=1000,
    n_test=1000,
    max_features = 2**16,
    n_jobs=None,
    verbose=False
):

    print("Start loading datasets")

    train_texts, train_labels = load_split(data_dir, human_source, computer_source, 'train', n=n_train)
    valid_texts, valid_labels = load_split(data_dir, human_source, computer_source, 'valid', n=n_valid)
    test_texts, test_labels = load_split(data_dir, human_source, computer_source, 'test', n=n_test)

    print("Datasets loaded")

    vect_name = f'tfidf_vect_{max_features}_feat.bin'
    model_name = f'tfidf_model_{max_features}_feat.bin'
    log_name = f'tfidf_model_{max_features}_feat.json'

    if has_model(model_dir, computer_source, vect_name):

        print(f"Load {vect_name} vect")
        vect = load_model(model_dir, computer_source, vect_name)

    else:

        print(f"Initialize {vect_name}")
        vect = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=max_features)
        print(f"Start training vect")
        vect.fit_transform(train_texts)
        print(f"Dump vect")
        dump_model(model_dir, computer_source, vect_name, vect)
        print(f"Vect dumped")

    valid_features = vect.transform(valid_texts)
    test_features = vect.transform(test_texts)


    if has_model(model_dir, computer_source, model_name):

        print(f"Load {model_name} model")
        model = load_model(model_dir, computer_source, model_name)

    else:

        train_features = vect.transform(train_texts)

        model = LogisticRegression(solver='liblinear')

        print("LogisticRegression initialized!")

        params = {'C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
        split = PredefinedSplit([-1]*n_train+[0]*n_valid)
        search = GridSearchCV(model, params, cv=split, n_jobs=n_jobs, verbose=verbose, refit=False)

        print("Start grid search!")

        search.fit(sparse.vstack([train_features, valid_features]), train_labels+valid_labels)
        model = model.set_params(**search.best_params_)

        print("Fit model!")

        model.fit(train_features, train_labels)
        dump_model(model_dir, computer_source, model_name, model)
        print(f"Model dumped")

    valid_accuracy = model.score(valid_features, valid_labels)*100.
    test_accuracy = model.score(test_features, test_labels)*100.

    data = {
        'source':computer_source,
        'n_train':n_train,
        'valid_accuracy':valid_accuracy,
        'test_accuracy':test_accuracy
    }
    print(data)
    json.dump(data, open(get_path(log_dir, computer_source, log_name), 'w'))

if __name__ == '__main__':
    fire.Fire(main)
