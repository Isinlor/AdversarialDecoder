import math
import pickle
import time

from AdvDecoder import decode, batch_decode
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipelineHF, IsFakePipelineSklearn

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(0)

# detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
# detector_model = RobertaForSequenceClassification.from_pretrained("./model/best-imporved")
# detector_model.to(0)

# classifier = IsFakePipelineHF(model=detector_model, tokenizer=detector_tokenizer, device=0)

classifier = IsFakePipelineSklearn(
    model=pickle.load(open('./model/sim.tfidf_model_65536_feat.bin', 'rb')),
    vectorizer=pickle.load(open('./model/sim.tfidf_vect_65536_feat.bin', 'rb'))
)

generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)

def benchmark(snippet, *args, **kargs):
    start = time.monotonic()
    result = snippet(*args, **kargs)
    end = time.monotonic()
    return (result, end - start)

summarize = lambda classifications: {
    "min": classifications.min(initial=1),
    "mean": classifications.mean(),
    "meanAbsDev": abs(classifications - classifications.mean()).mean(),
    "max": classifications.max(initial=0)
}

for step in reversed([4, 8, 16, 32]):
    for sequences in [4, 8, 12, 16, 32]:
        generate = lambda: batch_decode(
            batch=10,
            prompt="",
            generate_length=64,
            step=step,
            sequences_per_step=sequences,
            generator=generator,
            classifier=classifier
        )
        result, duration = benchmark(generate)
        summary = summarize(classifier.predict(result))
        print(f"Step {step}, sequences {sequences}: ", summary['mean'], duration)
