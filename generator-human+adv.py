import json
import os
import pickle

import fire
import jsonlines
from progressbar import progressbar

from AdvDecoder import decode

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipelineHF, IsFakePipelineSklearn

def _load_split(data_dir, name, split, n):
    path = os.path.join(data_dir, f'{name}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def main(lines, step=16, sequences_per_step=12, sequence_length=64):

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    detector_model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")

    model.to(0)
    detector_model.to(0)

    classifier = IsFakePipelineHF(model=detector_model, tokenizer=detector_tokenizer, device=0)

    # classifier = IsFakePipelineSklearn(
    #     model=pickle.load(open('./model/sim.tfidf_model_65536_feat.bin', 'rb')),
    #     vectorizer=pickle.load(open('./model/sim.tfidf_vect_65536_feat.bin', 'rb'))
    # )

    generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)

    human_texts = _load_split('./data', 'webtext', 'train', lines)

    for human_length in [32, 16, 8, 4]:
        with jsonlines.open(f"./data/human-prompt-{human_length}.test.jsonl", mode='a') as writer:
            for human_text in progressbar(human_texts):
                writer.write({
                    'text': decode(
                        prompt=tokenizer.decode(tokenizer.encode(human_text)[:human_length]),
                        step=step,
                        sequences_per_step=sequences_per_step,
                        generate_length=sequence_length,
                        generator=generator,
                        classifier=classifier
                    )
                })

if __name__ == '__main__':
    fire.Fire(main)