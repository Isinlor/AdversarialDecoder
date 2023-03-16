import fire
import jsonlines
from progressbar import progressbar

from AdvDecoder import decode

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipelineHF

def main(file, lines, sequences_per_step=12, sequence_length=64):

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model.to(0)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    model.to(0)

    generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)

    with jsonlines.open(file, mode='a') as writer:
        for _ in progressbar(range(lines // sequences_per_step)):
            sequences = generator.generate(
                prompt='',
                generate_length=sequence_length,
                num_return_sequences=sequences_per_step,
                do_sample=True,
                top_p=0.99,
                no_repeat_ngram_size=3
            )
            for text in sequences:
                writer.write({'text': text})

if __name__ == '__main__':
    fire.Fire(main)