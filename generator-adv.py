import pickle

import fire
import jsonlines
from progressbar import progressbar

from AdvDecoder import decode
from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipelineHF, IsFakePipelineSklearn


from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification




def main(file, lines, step=16, sequences_per_step=12, sequence_length=64):

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

    with jsonlines.open(file, mode='a') as writer:
        for _ in progressbar(range(lines)):
            writer.write({'text': decode(
                prompt="",
                step=step,
                sequences_per_step=sequences_per_step,
                generate_length=sequence_length,
                generator=generator,
                classifier=classifier
            )})

if __name__ == '__main__':
    fire.Fire(main)