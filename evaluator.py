import pickle

import fire
import jsonlines
import numpy as np
import transformers

from transformers import AutoTokenizer
from transformers.models.auto import AutoModelForSequenceClassification

from IsFakePipeline import IsFakePipelineHF, IsFakePipelineSklearn


def main(model, file, target, count = 500):

    target = int(target)

    transformers.logging.set_verbosity_error()

    if model == "roberta":
        detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
        detector_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
        detector_model.to(0)
        classifier = IsFakePipelineHF(model=detector_model, tokenizer=detector_tokenizer, device=0)
    elif model == "roberta-adv":
        detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
        detector_model = AutoModelForSequenceClassification.from_pretrained("./model/best-imporved")
        detector_model.to(0)
        classifier = IsFakePipelineHF(model=detector_model, tokenizer=detector_tokenizer, device=0)
    elif model == "tfidf-gpt2":
        classifier = IsFakePipelineSklearn(
            model=pickle.load(open('model/sim.tfidf_model_65536_feat.bin', 'rb')),
            vectorizer=pickle.load(open('model/sim.tfidf_vect_65536_feat.bin', 'rb'))
        )
    elif model == "tfidf-gpt-neo":
        classifier = IsFakePipelineSklearn(
            model=pickle.load(open('model/sim-neo.tfidf_model_65536_feat.bin', 'rb')),
            vectorizer=pickle.load(open('model/sim-neo.tfidf_vect_65536_feat.bin', 'rb'))
        )
    elif model == "tfidf-roberta-adv":
        classifier = IsFakePipelineSklearn(
            model=pickle.load(open('model/adv.tfidf_model_65536_feat.bin', 'rb')),
            vectorizer=pickle.load(open('model/adv.tfidf_vect_65536_feat.bin', 'rb'))
        )
    elif model == "tfidf-adv":
        classifier = IsFakePipelineSklearn(
            model=pickle.load(open('model/adv-tfidf.tfidf_model_65536_feat.bin', 'rb')),
            vectorizer=pickle.load(open('model/adv-tfidf.tfidf_vect_65536_feat.bin', 'rb'))
        )
    else:
        raise Exception(f"Wrong model {model}!")


    predictions = []
    with jsonlines.open(file) as reader:
        for sequence in reader:
            prediction = classifier.predict([sequence['text']])[0]
            predictions.append(prediction)
            if len(predictions) >= count:
                break

    predictions = np.array(predictions)
    correct = int(np.array(list(map(lambda prediction: int(target == round(prediction)), predictions))).sum())
    accuracy = correct / len(predictions)
    print(f"file={(file+',').ljust(30, ' ')}\ttarget={target},\tn={len(predictions)},\tcorrect={correct},\taccuracy={round(accuracy, 3)},\tmean={format(round(predictions.mean(), 3), '.3f')},\tstd={format(round(predictions.std(), 3), '.3f')}")

if __name__ == '__main__':
    fire.Fire(main)