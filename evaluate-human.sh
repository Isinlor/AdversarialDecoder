echo "TFIDF-GPT2"
python ./evaluator.py tfidf-gpt2 ./data/human-prompt-4.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/human-prompt-8.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/human-prompt-16.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/human-prompt-32.test.jsonl 1

echo "TFIDF-GPT-NEO"
python ./evaluator.py tfidf-gpt-neo ./data/human-prompt-4.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/human-prompt-8.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/human-prompt-16.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/human-prompt-32.test.jsonl 1

echo "RoBERTa"
python ./evaluator.py roberta ./data/human-prompt-4.test.jsonl 1
python ./evaluator.py roberta ./data/human-prompt-8.test.jsonl 1
python ./evaluator.py roberta ./data/human-prompt-16.test.jsonl 1
python ./evaluator.py roberta ./data/human-prompt-32.test.jsonl 1

echo "RoBERTa-Adv"
python ./evaluator.py roberta-adv ./data/human-prompt-4.test.jsonl 1
python ./evaluator.py roberta-adv ./data/human-prompt-8.test.jsonl 1
python ./evaluator.py roberta-adv ./data/human-prompt-16.test.jsonl 1
python ./evaluator.py roberta-adv ./data/human-prompt-32.test.jsonl 1