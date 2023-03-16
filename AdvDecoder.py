def decode(prompt, generate_length, step, sequences_per_step, generator, classifier):
    for i in range(0, generate_length, step):
        texts = generator.generate(prompt=prompt, generate_length=step, num_return_sequences=sequences_per_step, do_sample=True, top_p=0.99, no_repeat_ngram_size=3)
        prompt, _ = classifier.getLeastFake(texts)
    return prompt

def batch_decode(batch, prompt, generate_length, step, sequences_per_step, generator, classifier):
    return [decode(prompt, generate_length, step, sequences_per_step, generator, classifier) for _ in range(batch)]