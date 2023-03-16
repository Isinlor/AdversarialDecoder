import os
import json
import pickle

import fire
import numpy as np
from jsonlines import jsonlines
from scipy import sparse

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer


def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def main(
    filename,
    data_dir='./data'
):

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    normalized = os.path.join(data_dir, f'{filename}2.jsonl')
    file = os.path.join(data_dir, f'{filename}.jsonl')

    with jsonlines.open(normalized, mode='w') as writer:
        with jsonlines.open(file) as reader:
            for sequence in reader:
                encoded = tokenizer.encode(sequence['text'][:63*30])[:63]
                if len(encoded) < 63:
                    continue
                writer.write({'text': tokenizer.decode(encoded[:63])})

if __name__ == '__main__':
    fire.Fire(main)
