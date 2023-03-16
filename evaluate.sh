echo "TFIDF-GPT2"
python ./evaluator.py tfidf-gpt2 ./data/webtext.test.jsonl 0
python ./evaluator.py tfidf-gpt2 ./data/sim.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/adv.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/adv-tfidf.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/gpt-3-175.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/sim-neo.test.jsonl 1
python ./evaluator.py tfidf-gpt2 ./data/pile.test.jsonl 0

echo "TFIDF-GPT-NEO"
python ./evaluator.py tfidf-gpt-neo ./data/webtext.test.jsonl 0
python ./evaluator.py tfidf-gpt-neo ./data/sim.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/adv.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/adv-tfidf.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/gpt-3-175.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/sim-neo.test.jsonl 1
python ./evaluator.py tfidf-gpt-neo ./data/pile.test.jsonl 0

echo "RoBERTa"
python ./evaluator.py roberta ./data/webtext.test.jsonl 0
python ./evaluator.py roberta ./data/sim.test.jsonl 1
python ./evaluator.py roberta ./data/adv.test.jsonl 1
python ./evaluator.py roberta ./data/adv-tfidf.test.jsonl 1
python ./evaluator.py roberta ./data/gpt-3-175.test.jsonl 1
python ./evaluator.py roberta ./data/sim-neo.test.jsonl 1
python ./evaluator.py roberta ./data/pile.test.jsonl 0

echo "RoBERTa-Adv"
python ./evaluator.py roberta-adv ./data/webtext.test.jsonl 0
python ./evaluator.py roberta-adv ./data/sim.test.jsonl 1
python ./evaluator.py roberta-adv ./data/adv.test.jsonl 1
python ./evaluator.py roberta-adv ./data/adv-tfidf.test.jsonl 1
python ./evaluator.py roberta-adv ./data/gpt-3-175.test.jsonl 1
python ./evaluator.py roberta-adv ./data/sim-neo.test.jsonl 1
python ./evaluator.py roberta-adv ./data/pile.test.jsonl 0