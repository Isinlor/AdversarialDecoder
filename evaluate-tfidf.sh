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

echo "TFIDF-ROBERTA-ADV"
python ./evaluator.py tfidf-roberta-adv ./data/webtext.test.jsonl 0
python ./evaluator.py tfidf-roberta-adv ./data/sim.test.jsonl 1
python ./evaluator.py tfidf-roberta-adv ./data/adv.test.jsonl 1
python ./evaluator.py tfidf-roberta-adv ./data/adv-tfidf.test.jsonl 1
python ./evaluator.py tfidf-roberta-adv ./data/gpt-3-175.test.jsonl 1
python ./evaluator.py tfidf-roberta-adv ./data/sim-neo.test.jsonl 1
python ./evaluator.py tfidf-roberta-adv ./data/pile.test.jsonl 0

echo "TFIDF-ADV"
python ./evaluator.py tfidf-adv ./data/webtext.test.jsonl 0
python ./evaluator.py tfidf-adv ./data/sim.test.jsonl 1
python ./evaluator.py tfidf-adv ./data/adv.test.jsonl 1
python ./evaluator.py tfidf-adv ./data/adv-tfidf.test.jsonl 1
python ./evaluator.py tfidf-adv ./data/gpt-3-175.test.jsonl 1
python ./evaluator.py tfidf-adv ./data/sim-neo.test.jsonl 1
python ./evaluator.py tfidf-adv ./data/pile.test.jsonl 0