import random

import torch
import numpy as np
import os
import json
from transformers import Trainer, TrainingArguments
from datasets import load_metric

from AdvDecoder import decode, batch_decode

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipelineHF, IsFakePipelineSklearn


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def _load_split(data_dir, name, split, n=np.inf):
    path = os.path.join(data_dir, f'{name}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def load_split(tokenizer, data_dir, human, computer, split, n=np.inf):
    human_texts = _load_split(data_dir, human, split, n)
    computer_texts = _load_split(data_dir, computer, split, n)
    texts = tokenizer(human_texts + computer_texts, padding="max_length", truncation=True)
    labels = [0] * len(human_texts) + [1] * len(computer_texts)
    # return tokenizer(human_texts, padding="max_length", truncation=True), [1] * len(human_texts)
    # return tokenizer(computer_texts, padding="max_length", truncation=True), [0] * len(computer_texts)
    return texts, labels

from transformers import AutoTokenizer, RobertaForSequenceClassification
detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")

all_human_texts = _load_split('./data', 'webtext', 'train', 100000)

# train_dataset = MyDataset(*load_split(detector_tokenizer, './data', 'webtext', 'adv', 'train', 100))
# eval_dataset = MyDataset(*load_split(detector_tokenizer, './data', 'webtext', 'adv', 'valid', 100))

detector_model = RobertaForSequenceClassification.from_pretrained("./model/best")

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

generator_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
generator_model = GPT2LMHeadModel.from_pretrained("gpt2")
generator_model.to(0)

generator = BatchTextGenerationPipeline(model=generator_model, tokenizer=generator_tokenizer, device=0)
classifier = IsFakePipelineHF(model=detector_model, tokenizer=detector_tokenizer, device=0)

# print(classifier.predict(random.sample(all_human_texts, 1)).mean())
# print(classifier.predict(random.sample(all_human_texts, 100)).mean())
# print(classifier.predict(random.sample(all_human_texts, 100)).mean())
# print(classifier.predict(random.sample(all_human_texts, 100)).mean())
# print(classifier.predict(random.sample(all_human_texts, 100)).mean())
#

gen = _load_split('./data', 'sim', 'train', 9000)

# print(classifier.predict(random.sample(sim, 100)).mean())
# print(classifier.predict(random.sample(sim, 100)).mean())
# print(classifier.predict(random.sample(sim, 100)).mean())
# print(classifier.predict(random.sample(sim, 100)).mean())
# print(classifier.predict(random.sample(sim, 100)).mean())
# exit()

# human: 0, computer: 1

gen_validate = _load_split('./data', 'sim', 'valid', 150)
hum_validate = _load_split('./data', 'webtext', 'valid', 150)

best_error = 2
all_adv_computer_texts = []
for _ in range(50):

    print(f"epoch: {_}")

    new_adv_computer_samples = 10
    old_adv_computer_samples = 5
    gen_computer_samples = 5

    gen_computer_texts = random.sample(gen, gen_computer_samples)

    new_adv_computer_texts = batch_decode(
        batch=new_adv_computer_samples,
        prompt="",
        step=16,
        sequences_per_step=12,
        generate_length=64,
        generator=generator,
        classifier=classifier
    )

    if len(all_adv_computer_texts) > old_adv_computer_samples:
        old_computer_texts = random.sample(all_adv_computer_texts, old_adv_computer_samples)
    else:
        old_computer_texts = []

    all_adv_computer_texts += new_adv_computer_texts

    computer_texts = new_adv_computer_texts + old_computer_texts + gen_computer_texts
    human_texts = random.sample(all_human_texts, len(computer_texts))

    # print(np.array(map(lambda x: len(detector_tokenizer(computer_texts)))).mean())
    # print(np.array(map(lambda x: len(detector_tokenizer(human_texts)))).mean())
    # exit()

    # print("\nComputer: ----------------------------------------------------------------\n".join(computer_texts))
    # print("\nHuman: -------------------------------------------------------------------\n".join(human_texts))

    train_dataset = MyDataset(
        detector_tokenizer(computer_texts + human_texts, padding="max_length", truncation=True),
        [0] * len(computer_texts) + [1] * len(human_texts)
    )

    training_args = TrainingArguments(
        "test_trainer",
        # evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=20,
        seed=random.randint(0, 2**32 - 1),
        # do_eval=False,
        learning_rate=3e-6
    )

    trainer = Trainer(
        model=detector_model,
        args=training_args,
        train_dataset=train_dataset,
        # compute_metrics=compute_metrics,
    )

    computer_predictions = classifier.predict(computer_texts)
    human_predictions = classifier.predict(human_texts)

    computer_validate_predictions = classifier.predict(gen_validate)
    human_validate_predictions = classifier.predict(hum_validate)

    computer_all_predictions = np.concatenate((computer_predictions, computer_validate_predictions), axis=None)
    human_all_predictions = np.concatenate((human_predictions, human_validate_predictions), axis=None)
    error = human_all_predictions.mean() + (1 - computer_all_predictions.mean())

    print(
        "Adv new 1: ", "%.3f" % classifier.predict(new_adv_computer_texts).mean(),
        "Adv all 1: ", "%.3f" % classifier.predict(random.sample(all_adv_computer_texts, min(150, len(all_adv_computer_texts)))).mean() if len(old_computer_texts) else None,
        "Com 1: ", "%.3f" % computer_all_predictions.mean(),
        "Hum 0: ", "%.3f" % human_all_predictions.mean(),
        "Err: ", "%.3f" % error
    )

    if error < best_error:
        best_error = error
        trainer.save_model("./model/best")
        print("New best model")

    trainer.save_model("./model/latest")

    trainer.train()

    print(
        "Gen 1: ", "%.3f" % classifier.predict(gen_computer_texts).mean(),
        "Adv new 1: ", "%.3f" % classifier.predict(new_adv_computer_texts).mean(),
        "Adv old 1: ", "%.3f" % classifier.predict(old_computer_texts).mean() if len(old_computer_texts) else None,
        "Com 1: ", "%.3f" % classifier.predict(computer_texts).mean(),
        "Hum 0: ", "%.3f" % classifier.predict(human_texts).mean()
    )

    print("\n\n\n\n\n\n")