This code base can be used to reproduce experiments from the paper.

You can use `evaluate.sh` in order to evaluate many models on many datasets.

- `baseline.py` allows to train a baseline
- `traning.py` allows to adversarially train a model
- `search.py` allows for grid search on model hyperparams

The adversarial decoder consists of:

- `BatchTextGenerationPipeline.py`
- `IsFakePipeline.py`
- `AdvDecoder.py`

You can use it as so:

```python
    from AdvDecoder import decode
    from BatchTextGenerationPipeline import BatchTextGenerationPipeline
    from IsFakePipeline import IsFakePipelineHF, IsFakePipelineSklearn
    from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    detector_model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")
    
    model.to(0)
    detector_model.to(0)
    
    classifier = IsFakePipelineHF(model=detector_model, tokenizer=detector_tokenizer, device=0)
    generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
    
    decode(
        prompt="",
        step=step,
        sequences_per_step=sequences_per_step,
        generate_length=sequence_length,
        generator=generator,
        classifier=classifier
    )
```